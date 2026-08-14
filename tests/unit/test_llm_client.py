"""The client's accounting path: cache, budget, ledger, tiers -- no network."""

from __future__ import annotations

from typing import Any

import pytest

from ml_research_agent.config import BudgetConfig, Config, ObservabilityConfig
from ml_research_agent.errors import AgentFailure, BudgetExceeded, ConfigError
from ml_research_agent.llm.budget import BudgetTracker
from ml_research_agent.llm.cache import ResponseCache
from ml_research_agent.llm.client import FakeLLMClient, LLMClient, supports_sampling_params
from ml_research_agent.observability import logging as mra_logging
from ml_research_agent.observability.cost import CostLedger
from ml_research_agent.observability.logging import configure_logging
from ml_research_agent.observability.tracing import Tracer
from ml_research_agent.types import ModelTier

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]


def test_text_response_is_parsed(config: Config) -> None:
    client = FakeLLMClient(["hi there"], config=config)

    response = client.complete(messages=MESSAGES)

    assert response.text == "hi there"
    assert response.tool_calls == []
    assert response.tier == "standard"
    assert response.model == config.model_for("standard")
    assert response.cached is False


def test_tier_selects_the_configured_model(config: Config) -> None:
    client = FakeLLMClient(["a", "b", "c"], config=config)

    assert client.complete(messages=MESSAGES, tier=ModelTier.FAST).model == config.model_for("fast")
    assert client.complete(messages=MESSAGES, tier="deep").model == config.model_for("deep")
    assert client.complete(messages=MESSAGES).model == config.model_for("standard")


def test_cost_is_recorded_to_ledger_and_budget(config: Config) -> None:
    ledger = CostLedger()
    budget = BudgetTracker(BudgetConfig(), ledger=None)
    client = FakeLLMClient(
        [{"text": "ok", "input_tokens": 1_000_000, "output_tokens": 0}],
        config=config,
        ledger=ledger,
        budget=budget,
    )

    response = client.complete(messages=MESSAGES, phase="frame", agent="framer")

    assert response.cost_usd > 0
    assert ledger.total_usd == pytest.approx(response.cost_usd)
    assert ledger.by_phase() == {"frame": pytest.approx(response.cost_usd)}
    assert budget.spent_usd == pytest.approx(response.cost_usd)
    assert budget.tokens_for_phase("frame") == 1_000_000


def test_budget_stops_the_call_before_it_happens(config: Config) -> None:
    budget = BudgetTracker(BudgetConfig(usd_per_project=0.001))
    ledger = CostLedger()
    client = FakeLLMClient(["never reached"], config=config, budget=budget, ledger=ledger)

    with pytest.raises(BudgetExceeded) as excinfo:
        client.complete(messages=MESSAGES, phase="frame")

    assert client.calls == []
    # Refused before the spend: nothing booked to the budget or the ledger.
    assert budget.spent_usd == 0.0
    assert budget.tokens_for_phase("frame") == 0
    assert ledger.entries == []
    assert excinfo.value.retryable is False


def test_second_identical_call_is_served_from_cache(config: Config, tmp_path: Any) -> None:
    cache = ResponseCache(tmp_path / "llmcache")
    ledger = CostLedger()
    client = FakeLLMClient(["cached answer"], config=config, cache=cache, ledger=ledger)

    first = client.complete(messages=MESSAGES)
    second = client.complete(messages=MESSAGES)

    assert first.cached is False
    assert second.cached is True
    assert second.text == "cached answer"
    assert len(client.calls) == 1  # the script only had one response
    assert ledger.total_usd == first.cost_usd  # a cache hit costs nothing


def test_offline_without_a_cache_hit_is_a_config_error(config: Config) -> None:
    # The guard exists to stop a *network* call, so it is tested on the client
    # that would make one.
    client = LLMClient(config.with_overrides(offline=True))
    assert client.reaches_network

    with pytest.raises(ConfigError):
        client.complete(messages=MESSAGES)


def test_offline_does_not_block_the_scripted_client(config: Config) -> None:
    # A fake client opens no connection, so refusing it would only make every
    # offline end-to-end test impossible to write.
    client = FakeLLMClient(["scripted"], config=config.with_overrides(offline=True))
    assert client.reaches_network is False
    assert client.complete(messages=MESSAGES).text == "scripted"


def test_calls_are_traced(config: Config) -> None:
    tracer = Tracer(enabled=True)
    client = FakeLLMClient(["ok"], config=config, tracer=tracer)

    response = client.complete(messages=MESSAGES, phase="frame", agent="framer")

    span = tracer.export()[0]
    assert span["name"] == "llm.complete"
    assert span["attributes"]["phase"] == "frame"
    assert span["attributes"]["agent"] == "framer"
    assert span["attributes"]["model"] == response.model
    assert span["attributes"]["cost_usd"] == response.cost_usd
    assert span["attributes"]["stop_reason"] == response.stop_reason
    assert span["status"] == "ok"


def test_the_api_key_never_reaches_logs_traces_or_the_ledger(
    config: Config, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLAN §6: secrets live in ``.env`` only -- never in logs, redacted in traces."""
    secret = "sk-ant-test-must-never-be-recorded"
    monkeypatch.setenv(config.llm.api_key_env, secret)
    # Guard against a vacuous pass: the key really is reachable from the config.
    assert config.api_key == secret

    monkeypatch.setattr(mra_logging, "_sink", mra_logging._sink)  # restored on teardown
    configure_logging(ObservabilityConfig(log_level="DEBUG", console=False), log_dir=tmp_path)
    tracer = Tracer(enabled=True)
    ledger = CostLedger(tmp_path / "cost.jsonl")
    client = FakeLLMClient(["ok"], config=config, tracer=tracer, ledger=ledger)

    client.complete(messages=MESSAGES, phase="frame", agent="framer")

    log_text = (tmp_path / "mra.jsonl").read_text(encoding="utf-8")
    assert log_text.strip()  # something was actually logged
    assert secret not in log_text
    assert secret not in str(tracer.export())
    assert secret not in (tmp_path / "cost.jsonl").read_text(encoding="utf-8")
    assert secret not in str(ledger.summary())
    # The key is read from the environment on demand; it is not part of config state.
    assert secret not in str(config.model_dump(mode="json"))


def test_a_callable_script_sees_the_request(config: Config) -> None:
    def respond(**request: Any) -> str:
        return f"model={request['model']} n={len(request['messages'])}"

    client = FakeLLMClient(respond, config=config)

    assert client.complete(messages=MESSAGES).text.startswith("model=")


def test_running_out_of_scripted_responses_fails_fast(config: Config) -> None:
    client = FakeLLMClient([], config=config)

    with pytest.raises(ConfigError):
        client.complete(messages=MESSAGES)


def test_a_persistently_failing_provider_becomes_agent_failure(config: Config) -> None:
    def boom(**request: Any) -> str:
        raise RuntimeError("provider is down")

    fast = config.with_overrides(**{"llm.max_retries": 0})
    client = FakeLLMClient(boom, config=fast)

    with pytest.raises(AgentFailure) as excinfo:
        client.complete(messages=MESSAGES)
    assert excinfo.value.retryable is True


def test_fake_client_never_opens_a_provider_connection(config: Config) -> None:
    with pytest.raises(ConfigError):
        FakeLLMClient(["x"], config=config)._client()


@pytest.mark.parametrize(
    ("model", "allowed"),
    [
        ("claude-haiku-4-5-20251001", True),
        ("claude-opus-4-6", True),
        ("claude-opus-5", False),
        ("claude-sonnet-5", False),
    ],
)
def test_sampling_params_are_only_sent_where_they_are_accepted(model: str, allowed: bool) -> None:
    # The current Opus/Sonnet 5 generation rejects `temperature` with a 400.
    assert supports_sampling_params(model) is allowed
