"""Realistic input payloads, one per registered agent.

The golden-prompt tests call each agent's real ``prompt_variables()`` rather
than guessing at it, which means they need a valid typed payload for every
agent. Building them here keeps the test readable and gives future agent tests
a domain fixture set to reuse.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ml_research_agent.agents.analyst import AnalystInput
from ml_research_agent.agents.code_analyst import CodeAnalystInput
from ml_research_agent.agents.critic import CritiqueInput
from ml_research_agent.agents.curator import NoteInput, ScreeningInput
from ml_research_agent.agents.experiment_planner import PlannerInput
from ml_research_agent.agents.framer import FramerInput
from ml_research_agent.agents.implementer import ImplementerInput
from ml_research_agent.agents.runner_agent import TriageInput
from ml_research_agent.agents.scout import ScoutInput
from ml_research_agent.agents.synthesizer import SynthesisInput
from ml_research_agent.agents.writer import WriterInput
from ml_research_agent.types import (
    Author,
    Citation,
    CodeRepo,
    Comparison,
    DatasetRef,
    DecisionRule,
    EnvSpec,
    ExperimentSpec,
    FailureSignature,
    FalsifiableClaim,
    Hypothesis,
    Idea,
    LicenseCategory,
    LicenseInfo,
    MetricDef,
    MetricReport,
    MetricSummary,
    Note,
    Paper,
    Passage,
    Phase,
    Provenance,
    Recipe,
    RecipeStep,
    RepoMap,
    ResearchBrief,
    Result,
    RunRecord,
    RunStatus,
    Scale,
    SuccessCriterion,
    Variable,
    Verdict,
    VerdictStatus,
)

PAPER_KEY = "arxiv:2401.00001"


def idea() -> Idea:
    return Idea(
        text="Does label smoothing help small vision transformers at matched compute?",
        domain="computer vision",
        constraints=["one A100", "under 6 GPU-hours", "public datasets only"],
    )


def brief() -> ResearchBrief:
    return ResearchBrief(
        idea_id="idea_fixture",
        title="Label smoothing in compute-matched small ViTs",
        problem_statement=(
            "Label smoothing is standard in ViT recipes, but its benefit at small scale under "
            "a fixed compute budget has not been isolated from the other recipe changes."
        ),
        motivation="Small-scale practitioners copy large-scale recipes without evidence.",
        claims=[
            FalsifiableClaim(
                statement="Label smoothing improves top-1 on CIFAR-100 for ViT-Tiny.",
                falsifier="The mean top-1 delta across 3 seeds is within one standard deviation.",
                confidence=0.4,
            )
        ],
        success_criteria=[
            SuccessCriterion(
                metric="top1_accuracy",
                comparator=">=",
                threshold=0.01,
                dataset="cifar100",
                rationale="A point of top-1 is the smallest delta practitioners act on.",
            )
        ],
        assumptions=["The baseline recipe is correctly reproduced."],
        scope_limits=["No datasets beyond CIFAR-100", "No models above ViT-Small"],
        search_terms=["label smoothing", "vision transformer", "small-scale training"],
        related_areas=["regularization", "knowledge distillation"],
    )


def paper() -> Paper:
    return Paper(
        title="Rethinking Label Smoothing for Vision Transformers",
        abstract="We revisit label smoothing across model scales and report mixed results.",
        authors=[Author(name="Ada Lovelace"), Author(name="Alan Turing")],
        year=2024,
        arxiv_id="2401.00001",
        citation_count=137,
        tldr="Label smoothing helps at scale and hurts below 10M parameters.",
        code_urls=["https://github.com/example/ls-vit"],
        sources=["arxiv"],
    )


def provenance() -> Provenance:
    return Provenance(source=PAPER_KEY, locator="§4.2 p.6", quote="we observe a 0.4 point drop")


def note() -> Note:
    return Note(
        title="Rethinking Label Smoothing for Vision Transformers",
        summary="Label smoothing helps above 10M parameters and hurts below it on CIFAR-100.",
        paper_key=PAPER_KEY,
        task="image classification",
        method="label smoothing sweep across model scales",
        datasets=["cifar100", "imagenet-1k"],
        baselines=["cross-entropy"],
        metrics=[
            MetricReport(
                name="top1_accuracy",
                value=0.734,
                dataset="cifar100",
                split="test",
                method="label smoothing 0.1",
                provenance=provenance(),
            )
        ],
        compute="8 x A100 for 40 hours",
        limitations=["Only one seed per configuration."],
        relevance_to_brief="Directly contradicts the brief's claim at small scale.",
        confidence=0.7,
        provenance=[provenance()],
    )


def hypothesis() -> Hypothesis:
    return Hypothesis(
        brief_id="brief_fixture",
        statement="Label smoothing at 0.1 improves ViT-Tiny top-1 on CIFAR-100.",
        rationale="The prior work's negative result used a single seed and an unmatched budget.",
        prediction="Top-1 rises by at least 1 point at matched compute.",
        falsifier="The paired delta across 3 seeds is under 1 point or the sign is negative.",
        prior_confidence=0.45,
    )


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis_id="hyp_fixture",
        title="Label smoothing ablation on ViT-Tiny / CIFAR-100",
        description="Matched-compute comparison of label smoothing 0.1 against plain CE.",
        scale=Scale.SMOKE,
        independent_variables=[
            Variable(name="label_smoothing", values=[0.0, 0.1], description="Smoothing epsilon.")
        ],
        dependent_variables=[MetricDef(name="top1_accuracy", higher_is_better=True)],
        controls=["identical seed set", "identical LR schedule", "identical step budget"],
        dataset=DatasetRef(name="cifar100", split="test", version="v1"),
        baselines=["cross_entropy"],
        treatment="label_smoothing_0.1",
        seeds=[0, 1, 2],
        decision_rule=DecisionRule(
            metric="top1_accuracy",
            comparator=">=",
            threshold=0.01,
            relative_to_baseline=True,
            min_effect_size=0.005,
            max_p_value=0.05,
            min_seeds=3,
        ),
        stopping_rule="Stop after all seeds complete or the budget is exhausted.",
        budget_usd=4.0,
        max_runtime_minutes=25,
        env=EnvSpec(python_version="3.11", packages=["torch==2.3.0"]),
    )


def repo() -> CodeRepo:
    return CodeRepo(
        url="https://github.com/example/ls-vit",
        name="ls-vit",
        owner="example",
        commit="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        default_branch="main",
        stars=412,
        language="Python",
        description="Reference implementation for the label smoothing paper.",
        readme="# ls-vit\n\nTrain with `python train.py --config configs/vit_tiny.yaml`.",
        license=LicenseInfo(
            spdx_id="MIT", category=LicenseCategory.PERMISSIVE, permits_adaptation=True
        ),
        paper_keys=[PAPER_KEY],
        is_official=True,
    )


def repo_map(repo_id: str) -> RepoMap:
    return RepoMap(
        repo_id=repo_id,
        entrypoints=["train.py", "eval.py"],
        config_files=["configs/vit_tiny.yaml"],
        config_system="yaml",
        train_scripts=["train.py"],
        eval_scripts=["eval.py"],
        model_files=["models/vit.py"],
        data_files=["data/cifar.py"],
        dependencies=["torch>=2.0", "timm", "numpy"],
        python_version="3.11",
        hardware_assumptions=["single GPU", "16GB VRAM"],
        file_count=42,
        loc=6100,
    )


def recipe(repo_id: str) -> Recipe:
    return Recipe(
        repo_id=repo_id,
        paper_key=PAPER_KEY,
        summary="Train ViT-Tiny on CIFAR-100 with a configurable smoothing epsilon.",
        setup=[RecipeStep(command="pip install -r requirements.txt", description="Install deps.")],
        smoke=[
            RecipeStep(
                command="python train.py --config configs/vit_tiny.yaml --max-steps 50",
                description="50-step correctness check.",
                expected_artifacts=["runs/smoke/metrics.json"],
                timeout_seconds=600,
            )
        ],
        evaluate=[RecipeStep(command="python eval.py --checkpoint runs/smoke/last.pt")],
        config_knobs={"label_smoothing": "float, default 0.0", "seed": "int"},
        datasets=["cifar100"],
        reference_numbers=[
            MetricReport(
                name="top1_accuracy", value=0.712, dataset="cifar100", provenance=provenance()
            )
        ],
        gotchas=["The dataloader silently falls back to CPU when CUDA is unavailable."],
        env=EnvSpec(python_version="3.11", packages=["torch==2.3.0", "timm==1.0.3"]),
        confidence=0.6,
    )


def runs(spec_obj: ExperimentSpec) -> list[RunRecord]:
    return [
        RunRecord(
            spec_id=spec_obj.id,
            spec_hash=spec_obj.spec_hash,
            code_hash="c0de1234",
            env_hash="e0v1234",
            seed=seed,
            arm=arm,
            scale=Scale.SMOKE,
            status=RunStatus.COMPLETED,
            duration_seconds=310.5,
            cost_usd=0.4,
        )
        for arm in ("cross_entropy", "label_smoothing_0.1")
        for seed in (0, 1, 2)
    ]


def failed_run(spec_obj: ExperimentSpec) -> RunRecord:
    return RunRecord(
        spec_id=spec_obj.id,
        spec_hash=spec_obj.spec_hash,
        code_hash="c0de1234",
        env_hash="e0v1234",
        seed=1,
        arm="label_smoothing_0.1",
        scale=Scale.SMOKE,
        status=RunStatus.FAILED,
        duration_seconds=94.2,
        failure=FailureSignature.OOM,
        failure_detail="CUDA out of memory: tried to allocate 2.10 GiB",
        stdout_tail="step 120 | loss 3.91\nRuntimeError: CUDA out of memory",
    )


def _summary(arm: str, mean: float) -> MetricSummary:
    return MetricSummary(
        name="top1_accuracy",
        arm=arm,
        n=3,
        mean=mean,
        std=0.004,
        min=mean - 0.005,
        max=mean + 0.005,
        values=[mean - 0.005, mean, mean + 0.005],
    )


def result(spec_obj: ExperimentSpec, run_records: list[RunRecord]) -> Result:
    baseline = _summary("cross_entropy", 0.702)
    treatment = _summary("label_smoothing_0.1", 0.719)
    return Result(
        spec_id=spec_obj.id,
        spec_hash=spec_obj.spec_hash,
        run_ids=[r.id for r in run_records],
        summaries=[baseline, treatment],
        comparisons=[
            Comparison(
                metric="top1_accuracy",
                baseline_arm="cross_entropy",
                treatment_arm="label_smoothing_0.1",
                baseline=baseline,
                treatment=treatment,
                effect=0.017,
                relative_effect=0.0242,
                p_value=0.031,
                effect_size=1.9,
                n_seeds=3,
            )
        ],
        n_seeds=3,
        failed_runs=0,
        total_cost_usd=2.4,
        notes=["All six runs completed."],
    )


def verdict(result_obj: Result) -> Verdict:
    return Verdict(
        hypothesis_id="hyp_fixture",
        result_id=result_obj.id,
        status=VerdictStatus.SUPPORTED,
        reasoning="The paired delta cleared the pre-registered threshold across three seeds.",
        decision_rule_statement="Refuted unless top1_accuracy improves by >= 0.01.",
        rule_fired=True,
        seed_variance_note="Seed spread was 0.4 points, well inside the observed effect.",
        threats_to_validity=["Only one dataset", "Smoke scale only"],
        confidence=0.6,
    )


def passages() -> list[Passage]:
    return [
        Passage(
            doc_id=PAPER_KEY,
            text="On CIFAR-100 we observe a 0.4 point drop for models under 10M parameters.",
            section="4.2 Results",
            page=6,
            order=12,
            token_count=180,
        ),
        Passage(
            doc_id=PAPER_KEY,
            text="All experiments use a single seed due to compute constraints.",
            section="5 Limitations",
            page=8,
            order=19,
            token_count=90,
        ),
    ]


def _payloads() -> dict[str, BaseModel]:
    brief_obj = brief()
    spec_obj = spec()
    repo_obj = repo()
    run_records = runs(spec_obj)
    result_obj = result(spec_obj, run_records)

    return {
        "framer": FramerInput(idea=idea(), context="The team has one GPU for two days."),
        "scout": ScoutInput(brief=brief_obj, known_keys=[PAPER_KEY]),
        "screener": ScreeningInput(brief=brief_obj, candidates=[paper()]),
        "curator": NoteInput(brief=brief_obj, paper=paper(), passages=passages()),
        "code_analyst": CodeAnalystInput(
            repo=repo_obj,
            repo_map=repo_map(repo_obj.id),
            paper_key=PAPER_KEY,
            paper_title=paper().title,
            target_numbers=[
                MetricReport(
                    name="top1_accuracy",
                    value=0.712,
                    dataset="cifar100",
                    provenance=provenance(),
                )
            ],
            file_excerpts={"train.py": "def main():\n    cfg = load_config()\n"},
        ),
        "synthesizer": SynthesisInput(
            brief=brief_obj,
            notes=[note()],
            contradictions=[(PAPER_KEY, "arxiv:2402.09999")],
            coverage_gaps=[{"method": "label smoothing", "dataset": "tiny-imagenet", "n": 0}],
        ),
        "experiment_planner": PlannerInput(
            hypothesis=hypothesis(),
            baselines=["cross_entropy"],
            benchmarks=["cifar100"],
            recipes=[recipe(repo_obj.id)],
            budget_usd=4.0,
            default_seeds=3,
            scale=Scale.SMOKE,
        ),
        "implementer": ImplementerInput(
            spec=spec_obj,
            recipe=recipe(repo_obj.id),
            workspace_path="runs/spec_fixture",
            existing_files=["train.py", "configs/vit_tiny.yaml"],
            notes=["Reuse the reference dataloader unchanged."],
        ),
        "runner": TriageInput(
            spec=spec_obj,
            run=failed_run(spec_obj),
            log_tail="step 120 | loss 3.91\nRuntimeError: CUDA out of memory",
            attempt=1,
            max_attempts=3,
        ),
        "analyst": AnalystInput(
            spec=spec_obj,
            runs=run_records,
            result=result_obj,
            rule_status=VerdictStatus.SUPPORTED,
            rule_detail="effect=0.017 threshold=0.01 p=0.031 seeds=3",
        ),
        "critic": CritiqueInput(
            phase=Phase.DESIGN,
            target_id=spec_obj.id,
            artifact=spec_obj.model_dump(mode="json"),
            context={"brief": brief_obj.title, "kb_papers": 34},
        ),
        "writer": WriterInput(
            brief=brief_obj,
            synthesis={"settled": ["label smoothing helps at scale"], "gaps": ["small-scale ViTs"]},
            results=[result_obj],
            verdicts=[verdict(result_obj)],
            available_citations=[
                Citation(key=PAPER_KEY, label="Lovelace et al., 2024", kind="paper"),
                Citation(key=run_records[0].id, label="run 0", kind="run"),
            ],
            followups=["Repeat at ViT-Small scale."],
            budget_note="Spent $2.40 of a $50 ceiling.",
        ),
    }


PAYLOADS: dict[str, BaseModel] = _payloads()


def payload_for(agent_name: str) -> Any:
    if agent_name not in PAYLOADS:
        raise KeyError(
            f"no fixture payload for agent {agent_name!r}; add one to tests/unit/agent_payloads.py"
        )
    return PAYLOADS[agent_name]


__all__ = ["PAYLOADS", "payload_for"]
