"""The response cache must be deterministic, or replays and prompt tests aren't."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_research_agent.llm.cache import ResponseCache

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "frame this idea"}]
PARAMS: dict[str, Any] = {"max_tokens": 1024, "temperature": 0.0, "system": "be terse"}


def test_key_is_stable_across_instances(tmp_path: Path) -> None:
    first = ResponseCache(tmp_path).key(model="m", messages=MESSAGES, params=PARAMS)
    second = ResponseCache(tmp_path).key(model="m", messages=MESSAGES, params=PARAMS)

    assert first == second


def test_key_changes_with_model_messages_and_params(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    base = cache.key(model="m", messages=MESSAGES, params=PARAMS)

    assert cache.key(model="other", messages=MESSAGES, params=PARAMS) != base
    assert (
        cache.key(model="m", messages=[{"role": "user", "content": "other"}], params=PARAMS) != base
    )
    assert cache.key(model="m", messages=MESSAGES, params={**PARAMS, "temperature": 1.0}) != base


def test_a_changed_param_misses_the_cache(tmp_path: Path) -> None:
    """Key inequality is not enough -- the changed request must actually miss."""
    cache = ResponseCache(tmp_path)
    key = cache.key(model="m", messages=MESSAGES, params=PARAMS)
    cache.set(key, {"text": "answer for the original request"})

    variants: list[dict[str, Any]] = [
        {"max_tokens": 2048},
        {"temperature": 1.0},
        {"system": "be verbose"},
        {"stop_sequences": ["STOP"]},
        {"tool_choice": {"type": "tool", "name": "emit"}},
    ]
    for variant in variants:
        other = cache.key(model="m", messages=MESSAGES, params={**PARAMS, **variant})
        assert other != key, f"{variant} must change the key"
        assert cache.get(other) is None, f"{variant} must miss the cache"

    assert cache.get(cache.key(model="other", messages=MESSAGES, params=PARAMS)) is None
    assert cache.get(key) == {"text": "answer for the original request"}


def test_different_prompts_never_collide(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    prompts = ["a", "b", "a b", "ab", "", "a\nb"]
    keys = {
        p: cache.key(model="m", messages=[{"role": "user", "content": p}], params=PARAMS)
        for p in prompts
    }

    assert len(set(keys.values())) == len(prompts)
    for prompt, key in keys.items():
        cache.set(key, {"text": prompt})
    for prompt, key in keys.items():
        assert cache.get(key) == {"text": prompt}


def test_irrelevant_params_do_not_change_the_key(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    base = cache.key(model="m", messages=MESSAGES, params=PARAMS)

    # Timeouts and correlation ids do not change the sampled text; if they
    # participated in the key nothing would ever hit.
    noisy = {**PARAMS, "timeout": 30, "request_id": "abc123"}
    assert cache.key(model="m", messages=MESSAGES, params=noisy) == base


def test_set_then_get_roundtrips(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache.key(model="m", messages=MESSAGES, params=PARAMS)

    assert cache.get(key) is None
    cache.set(key, {"text": "hello", "input_tokens": 3, "output_tokens": 1})

    assert cache.get(key) == {"text": "hello", "input_tokens": 3, "output_tokens": 1}
    assert cache.stats["hits"] == 1


def test_disabled_cache_never_stores_or_serves(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path, enabled=False)
    key = cache.key(model="m", messages=MESSAGES, params=PARAMS)
    cache.set(key, {"text": "hello"})

    assert cache.get(key) is None
    assert cache.stats["enabled"] is False


def test_survives_a_new_process(tmp_path: Path) -> None:
    key = ResponseCache(tmp_path).key(model="m", messages=MESSAGES, params=PARAMS)
    ResponseCache(tmp_path).set(key, {"text": "persisted"})

    assert ResponseCache(tmp_path).get(key) == {"text": "persisted"}
