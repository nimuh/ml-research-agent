"""Core domain models shared by every layer (pydantic).

Idea, ResearchBrief, Question, Paper, Author, Venue, CodeRepo, Artifact,
Note, Claim, Evidence, Hypothesis, ExperimentSpec, RunRecord, Metric,
Result, Verdict, ReportSection. Every model carries a stable id and a
provenance field so any downstream claim can be traced to its source.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .utils.hashing import hash_obj

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

Score = Annotated[float, Field(ge=0.0, le=1.0)]


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Stable, sortable-enough, human-greppable id: ``paper_9f2c1a4b``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Phase(StrEnum):
    """The ten phases of the research loop, in order."""

    FRAME = "frame"
    SURVEY = "survey"
    CURATE = "curate"
    GROUND = "ground"
    SYNTHESIZE = "synthesize"
    DESIGN = "design"
    IMPLEMENT = "implement"
    RUN = "run"
    ANALYZE = "analyze"
    REPORT = "report"


PHASE_ORDER: tuple[Phase, ...] = (
    Phase.FRAME,
    Phase.SURVEY,
    Phase.CURATE,
    Phase.GROUND,
    Phase.SYNTHESIZE,
    Phase.DESIGN,
    Phase.IMPLEMENT,
    Phase.RUN,
    Phase.ANALYZE,
    Phase.REPORT,
)


class ModelTier(StrEnum):
    """Cost policy lives in the router; agents ask for a tier, never a model id."""

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class Scale(StrEnum):
    """The scale ladder. Nothing launches at ``main`` before ``smoke`` passes."""

    SMOKE = "smoke"
    SMALL = "small"
    MAIN = "main"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    SKIPPED = "skipped"


class FailureSignature(StrEnum):
    """Categorized failure modes -- each carries its own retry policy.

    Retrying an OOM with the same batch size is not a strategy, which is why
    the runner must name the failure before deciding what to do about it.
    """

    OOM = "oom"
    NAN = "nan"
    HANG = "hang"
    DIVERGENCE = "divergence"
    SILENT_NOOP = "silent_noop"
    DEPENDENCY = "dependency"
    DATA_MISSING = "data_missing"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    UNKNOWN = "unknown"


class VerdictStatus(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class EvidenceKind(StrEnum):
    PASSAGE = "passage"
    RUN = "run"
    FIGURE = "figure"
    TABLE = "table"
    CODE = "code"


class ClaimRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    REPRODUCES = "reproduces"
    CITES = "cites"
    USES = "uses"


class LicenseCategory(StrEnum):
    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    NONCOMMERCIAL = "noncommercial"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class ScreeningDecisionKind(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    MAYBE = "maybe"


# ---------------------------------------------------------------------------
# Base model + provenance
# ---------------------------------------------------------------------------


class MRAModel(BaseModel):
    """Base for every domain model.

    ``extra="forbid"`` is deliberate: an LLM inventing a field should fail
    validation and trigger the repair loop rather than silently drop data.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class Provenance(MRAModel):
    """Where a piece of information came from, precisely enough to re-check it.

    ``source`` is a URI-ish identifier (``arxiv:2401.00001``, ``run:abc123``,
    ``agent:framer@v1``, ``file:workspace/kb/raw/ab/abcd.pdf``) and ``locator``
    narrows it to a passage (``§3.2 p.5``, ``table 2``, ``lines 40-58``).
    """

    source: str
    locator: str | None = None
    quote: str | None = Field(default=None, description="Verbatim snippet backing the assertion.")
    retrieved_at: datetime = Field(default_factory=utcnow)
    confidence: Score = 1.0

    def __str__(self) -> str:
        return f"{self.source}#{self.locator}" if self.locator else self.source


class Entity(MRAModel):
    """A domain object with a stable id, a timestamp, and traceable origins."""

    id: str
    created_at: datetime = Field(default_factory=utcnow)
    provenance: list[Provenance] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    def with_provenance(self, *items: Provenance) -> Self:
        return self.model_copy(update={"provenance": [*self.provenance, *items]})

    @property
    def is_traceable(self) -> bool:
        return bool(self.provenance)


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


class Idea(Entity):
    """The one-sentence input that starts a project."""

    id: str = Field(default_factory=lambda: new_id("idea"))
    text: str
    domain: str | None = None
    constraints: list[str] = Field(default_factory=list)


class SuccessCriterion(MRAModel):
    """A measurable bar, stated before any work happens."""

    metric: str
    comparator: Literal[">", ">=", "<", "<=", "=", "!="]
    threshold: float
    dataset: str | None = None
    rationale: str | None = None

    def evaluate(self, value: float) -> bool:
        match self.comparator:
            case ">":
                return value > self.threshold
            case ">=":
                return value >= self.threshold
            case "<":
                return value < self.threshold
            case "<=":
                return value <= self.threshold
            case "=":
                return value == self.threshold
            case "!=":
                return value != self.threshold


class FalsifiableClaim(MRAModel):
    """An assertion paired with what would prove it wrong.

    A claim without a falsifier is an opinion; the Framer's exit criterion is
    that at least one of these exists.
    """

    statement: str
    falsifier: str = Field(description="The observation that would refute this claim.")
    confidence: Score = 0.5


class ResearchBrief(Entity):
    """The Framer's output: a vague idea turned into something testable."""

    id: str = Field(default_factory=lambda: new_id("brief"))
    idea_id: str
    title: str
    problem_statement: str
    motivation: str = ""
    claims: list[FalsifiableClaim] = Field(min_length=1)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    scope_limits: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    related_areas: list[str] = Field(default_factory=list)

    @property
    def query_text(self) -> str:
        """Dense text used for semantic ranking against candidate papers."""
        return " ".join(
            [
                self.title,
                self.problem_statement,
                *(c.statement for c in self.claims),
                *self.search_terms,
            ]
        )


class Question(Entity):
    """A researchable sub-question derived from the brief."""

    id: str = Field(default_factory=lambda: new_id("q"))
    brief_id: str
    text: str
    kind: Literal["empirical", "conceptual", "reproduction", "ablation"] = "empirical"
    priority: Score = 0.5
    status: Literal["open", "answered", "abandoned"] = "open"
    answer: str | None = None


# ---------------------------------------------------------------------------
# Literature
# ---------------------------------------------------------------------------


class Author(MRAModel):
    name: str
    author_id: str | None = None
    affiliation: str | None = None


class Venue(MRAModel):
    name: str
    year: int | None = None
    kind: Literal["conference", "journal", "workshop", "preprint", "unknown"] = "unknown"


class PaperSection(MRAModel):
    """One parsed section, kept separate so citations can carry a locator."""

    title: str
    text: str
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None
    order: int = 0

    @property
    def locator(self) -> str:
        if self.page_start is not None:
            return f"§{self.title} p.{self.page_start}"
        return f"§{self.title}"


class Paper(Entity):
    """A candidate or ingested paper.

    Ids from several sources coexist; dedup merges records into the richest one
    while keeping every source link.
    """

    id: str = Field(default_factory=lambda: new_id("paper"))
    title: str
    abstract: str = ""
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    venue: Venue | None = None

    arxiv_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    openreview_id: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)

    url: str | None = None
    pdf_url: str | None = None
    code_urls: list[str] = Field(default_factory=list)

    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    tldr: str | None = None

    sources: list[str] = Field(default_factory=list, description="Adapters that returned it.")
    references: list[str] = Field(default_factory=list, description="Paper keys this cites.")
    citations: list[str] = Field(default_factory=list, description="Paper keys citing this.")

    sections: list[PaperSection] = Field(default_factory=list)
    full_text_hash: str | None = None
    raw_path: str | None = None

    @property
    def key(self) -> str:
        """The strongest stable identifier available, for dedup and citation."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.s2_id:
            return f"s2:{self.s2_id}"
        if self.openreview_id:
            return f"openreview:{self.openreview_id}"
        return f"title:{hash_obj(self.title.lower())}"

    @property
    def citation_label(self) -> str:
        first = self.authors[0].name.split()[-1] if self.authors else "Anon"
        return f"{first} et al., {self.year}" if self.year else f"{first} et al."

    @property
    def has_full_text(self) -> bool:
        return bool(self.sections)


class RankedPaper(MRAModel):
    """A paper plus the cheap ranking signal that ordered it.

    Ranking is a *filter*, never a relevance judgment -- that separation is why
    retrieval recall and screening precision can be tuned independently.
    """

    paper: Paper
    score: float
    components: dict[str, float] = Field(default_factory=dict)
    rank: int = 0


class ScreeningDecision(MRAModel):
    """The Curator's include/exclude call, with its reason recorded either way."""

    paper_key: str
    decision: ScreeningDecisionKind
    reason: str
    relevance: Score = 0.0


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class Passage(MRAModel):
    """A retrievable chunk with enough locators to cite precisely."""

    id: str = Field(default_factory=lambda: new_id("psg"))
    doc_id: str
    text: str
    section: str | None = None
    page: int | None = None
    order: int = 0
    token_count: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def locator(self) -> str:
        bits = [b for b in (self.section, f"p.{self.page}" if self.page else None) if b]
        return " ".join(bits) or f"chunk {self.order}"

    def to_provenance(self, *, quote: str | None = None) -> Provenance:
        return Provenance(source=self.doc_id, locator=self.locator, quote=quote or self.text[:280])


class SearchHit(MRAModel):
    """One retrieval result. Everything agents read from the KB arrives as this."""

    passage: Passage
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    citation: str = ""


class MetricReport(MRAModel):
    """A headline number extracted from a paper, with the passage that stated it."""

    name: str
    value: float | None = None
    unit: str | None = None
    dataset: str | None = None
    split: str | None = None
    method: str | None = None
    provenance: Provenance


class Note(Entity):
    """The KB's atomic unit: a structured reading of one paper.

    Every substantive field must be backed by a passage locator -- a note whose
    claims float free of the source is rejected at ingest.
    """

    id: str = Field(default_factory=lambda: new_id("note"))
    type: Literal["paper", "concept", "method", "comparison", "todo"] = "paper"
    paper_key: str | None = None
    title: str
    summary: str

    task: str | None = None
    method: str | None = None
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[MetricReport] = Field(default_factory=list)
    compute: str | None = None
    limitations: list[str] = Field(default_factory=list)
    relevance_to_brief: str | None = None
    confidence: Score = 0.5

    claims: list[str] = Field(default_factory=list, description="Claim ids asserted by this note.")
    sources: list[str] = Field(default_factory=list)
    supersedes: str | None = None
    added_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _require_provenance(self) -> Self:
        if self.type == "paper" and not self.provenance:
            raise ValueError("a paper note without passage-level provenance is not admissible")
        return self


class Evidence(MRAModel):
    """Links a claim to a paper passage *or* a run id. Nothing else counts."""

    kind: EvidenceKind
    provenance: Provenance
    supports: bool = True
    strength: Score = 0.5
    note: str | None = None


class Claim(Entity):
    """An assertion of the form "X improves Y under Z".

    ``contradicts`` edges between claims are a first-class research signal --
    a contested result is a candidate replication, not a data-quality bug.
    """

    id: str = Field(default_factory=lambda: new_id("claim"))
    statement: str
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    conditions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    supports_claims: list[str] = Field(default_factory=list)
    confidence: Score = 0.5

    @property
    def is_supported(self) -> bool:
        return any(e.supports for e in self.evidence)

    @property
    def is_contested(self) -> bool:
        return bool(self.contradicts) or (
            any(e.supports for e in self.evidence) and any(not e.supports for e in self.evidence)
        )


class GraphEdge(MRAModel):
    """A typed edge in the knowledge graph."""

    source: str
    target: str
    relation: ClaimRelation
    weight: float = 1.0
    provenance: Provenance | None = None


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


class LicenseInfo(MRAModel):
    spdx_id: str | None = None
    category: LicenseCategory = LicenseCategory.UNKNOWN
    permits_adaptation: bool = False
    notes: str | None = None
    provenance: Provenance | None = None


class CodeRepo(Entity):
    """A candidate reference implementation, pinned to a commit.

    ``local_path`` being set means the repo was *fetched*, never that anything
    in it was executed -- execution only ever happens inside the sandbox.
    """

    id: str = Field(default_factory=lambda: new_id("repo"))
    url: str
    name: str
    owner: str | None = None
    commit: str | None = None
    default_branch: str | None = None
    stars: int = 0
    forks: int = 0
    last_commit_at: datetime | None = None
    language: str | None = None
    description: str | None = None
    readme: str | None = None
    license: LicenseInfo = Field(default_factory=LicenseInfo)
    paper_keys: list[str] = Field(default_factory=list)
    is_official: bool = False
    fidelity_score: Score = 0.0
    local_path: str | None = None
    code_hash: str | None = None


class RepoMap(MRAModel):
    """Static analysis output: what the repo *is*, before anything runs."""

    repo_id: str
    entrypoints: list[str] = Field(default_factory=list)
    config_files: list[str] = Field(default_factory=list)
    config_system: str | None = None
    train_scripts: list[str] = Field(default_factory=list)
    eval_scripts: list[str] = Field(default_factory=list)
    model_files: list[str] = Field(default_factory=list)
    data_files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    python_version: str | None = None
    hardware_assumptions: list[str] = Field(default_factory=list)
    file_count: int = 0
    loc: int = 0
    notes: list[str] = Field(default_factory=list)


class EnvSpec(MRAModel):
    """Enough to rebuild the environment deterministically."""

    python_version: str = "3.11"
    cuda_version: str | None = None
    packages: list[str] = Field(default_factory=list)
    lockfile_path: str | None = None
    container_image: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)

    @property
    def env_hash(self) -> str:
        return hash_obj(
            {
                "python": self.python_version,
                "cuda": self.cuda_version,
                "packages": sorted(self.packages),
                "image": self.container_image,
            }
        )


class RecipeStep(MRAModel):
    command: str
    description: str = ""
    working_dir: str | None = None
    expected_artifacts: list[str] = Field(default_factory=list)
    timeout_seconds: int = 1800


class Recipe(Entity):
    """The runnable distillation of a repo: what to type, and what should happen.

    A repo without a Recipe is a citation. A repo with one is an experiment.
    """

    id: str = Field(default_factory=lambda: new_id("recipe"))
    repo_id: str
    paper_key: str | None = None
    summary: str = ""
    setup: list[RecipeStep] = Field(default_factory=list)
    smoke: list[RecipeStep] = Field(default_factory=list)
    train: list[RecipeStep] = Field(default_factory=list)
    evaluate: list[RecipeStep] = Field(default_factory=list)
    config_knobs: dict[str, str] = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)
    reference_numbers: list[MetricReport] = Field(default_factory=list)
    gotchas: list[str] = Field(default_factory=list)
    env: EnvSpec = Field(default_factory=EnvSpec)
    confidence: Score = 0.3

    @property
    def is_runnable(self) -> bool:
        return bool(self.smoke or self.train or self.evaluate)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class Hypothesis(Entity):
    """A prediction plus, mandatorily, what would falsify it."""

    id: str = Field(default_factory=lambda: new_id("hyp"))
    brief_id: str
    question_id: str | None = None
    statement: str
    rationale: str
    prediction: str
    falsifier: str
    prior_confidence: Score = 0.5
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "testing", "supported", "refuted", "inconclusive", "escalate"] = "open"
    rounds: int = Field(default=0, description="How many DESIGN->ANALYZE cycles it has consumed.")
    """``escalate`` is terminal for the automated loop: two inconclusive rounds
    on one hypothesis go to a human rather than to more compute."""


class Variable(MRAModel):
    name: str
    values: list[Any] = Field(default_factory=list)
    description: str = ""


class DatasetRef(MRAModel):
    name: str
    split: str = "test"
    version: str | None = None
    url: str | None = None
    size: str | None = None
    license: LicenseInfo | None = None


class MetricDef(MRAModel):
    name: str
    higher_is_better: bool = True
    unit: str | None = None
    description: str = ""


class DecisionRule(MRAModel):
    """The pre-registration. Written before compute is spent, evaluated after.

    ``supported`` requires the effect to clear ``threshold`` in the right
    direction *and* to survive the seed-variance guard; anything else is
    inconclusive or refuted. This is what makes a negative result a finding.
    """

    metric: str
    comparator: Literal[">", ">=", "<", "<="]
    threshold: float
    relative_to_baseline: bool = True
    min_effect_size: float = 0.0
    max_p_value: float | None = 0.05
    min_seeds: int = 3
    statement: str = ""

    @model_validator(mode="after")
    def _require_statement(self) -> Self:
        if not self.statement:
            direction = "improves" if self.comparator in (">", ">=") else "reduces"
            rel = " over the baseline" if self.relative_to_baseline else ""
            object.__setattr__(
                self,
                "statement",
                f"Refuted unless {self.metric} {direction}{rel} by {self.comparator} "
                f"{self.threshold} across >= {self.min_seeds} seeds.",
            )
        return self

    def evaluate(
        self, effect: float, *, p_value: float | None = None, seeds: int = 0
    ) -> VerdictStatus:
        """Apply the rule literally -- no judgement, no rationalizing."""
        if seeds < self.min_seeds:
            return VerdictStatus.INCONCLUSIVE
        if self.max_p_value is not None and (p_value is None or p_value > self.max_p_value):
            return VerdictStatus.INCONCLUSIVE
        if abs(effect) < self.min_effect_size:
            return VerdictStatus.REFUTED
        match self.comparator:
            case ">":
                passed = effect > self.threshold
            case ">=":
                passed = effect >= self.threshold
            case "<":
                passed = effect < self.threshold
            case "<=":
                passed = effect <= self.threshold
        return VerdictStatus.SUPPORTED if passed else VerdictStatus.REFUTED


class ExperimentSpec(Entity):
    """A pre-registered experiment. Hashable: ``spec_hash`` identifies it.

    Rejected at DESIGN if it has no decision rule, when
    ``require_preregistered_decision_rule`` is on.
    """

    id: str = Field(default_factory=lambda: new_id("spec"))
    hypothesis_id: str
    title: str
    description: str = ""
    scale: Scale = Scale.SMOKE

    independent_variables: list[Variable] = Field(default_factory=list)
    dependent_variables: list[MetricDef] = Field(min_length=1)
    controls: list[str] = Field(default_factory=list)
    dataset: DatasetRef
    baselines: list[str] = Field(default_factory=list)
    treatment: str = ""

    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    decision_rule: DecisionRule
    stopping_rule: str = ""
    budget_usd: float = 5.0
    max_runtime_minutes: int = 30
    recipe_id: str | None = None
    code_ref: str | None = None
    env: EnvSpec = Field(default_factory=EnvSpec)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("seeds")
    @classmethod
    def _seeds_nonempty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("an experiment with no seeds cannot be analyzed")
        return sorted(set(value))

    @property
    def spec_hash(self) -> str:
        """Identity of the *experiment*, not of this object.

        Deliberately excludes ids, timestamps, provenance and budget: rerunning
        the same design under a new id must land on the same hash.
        """
        return hash_obj(
            {
                "title": self.title,
                "scale": self.scale.value,
                "ivs": [v.model_dump(mode="json") for v in self.independent_variables],
                "dvs": [m.model_dump(mode="json") for m in self.dependent_variables],
                "controls": sorted(self.controls),
                "dataset": self.dataset.model_dump(mode="json"),
                "baselines": sorted(self.baselines),
                "treatment": self.treatment,
                "seeds": self.seeds,
                "decision_rule": self.decision_rule.model_dump(mode="json"),
                "parameters": self.parameters,
            }
        )

    @property
    def arms(self) -> list[str]:
        """Baselines plus the treatment -- every arm that must actually run."""
        return [*self.baselines, self.treatment] if self.treatment else list(self.baselines)


class Artifact(MRAModel):
    """A file a run produced, addressed by content hash."""

    name: str
    path: str
    kind: Literal["log", "metrics", "checkpoint", "figure", "table", "config", "other"] = "other"
    size_bytes: int = 0
    content_hash: str | None = None


class Metric(MRAModel):
    """One measured value from one run."""

    name: str
    value: float
    step: int | None = None
    split: str | None = None
    arm: str | None = None


class RunRecord(Entity):
    """``(spec_hash, code_hash, env_hash, seed)`` -> what happened.

    Those four hashes are the reproducibility contract: same tuple, same
    metrics, or :class:`IrreproducibleRun`.
    """

    id: str = Field(default_factory=lambda: new_id("run"))
    spec_id: str
    spec_hash: str
    code_hash: str = ""
    env_hash: str = ""
    seed: int = 0
    arm: str = "treatment"
    scale: Scale = Scale.SMOKE

    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float = 0.0

    metrics: list[Metric] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    log_path: str | None = None
    stdout_tail: str = ""
    failure: FailureSignature | None = None
    failure_detail: str | None = None
    retry_of: str | None = None
    cost_usd: float = 0.0
    workspace_path: str | None = None

    @property
    def repro_key(self) -> tuple[str, str, str, str, int]:
        """The reproducibility contract, including the arm.

        The plan states the contract as ``(spec_hash, code_hash, env_hash,
        seed)``, but a spec runs several arms at the same seed. Without the arm
        the baseline and the treatment share a key and every experiment looks
        irreproducible against itself.
        """
        return (self.spec_hash, self.code_hash, self.env_hash, self.arm, self.seed)

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.COMPLETED

    def metric(self, name: str) -> float | None:
        """Final value of a metric (last step wins)."""
        values = [m for m in self.metrics if m.name == name]
        return values[-1].value if values else None


class MetricSummary(MRAModel):
    """Aggregation across seeds. Spread is reported, never hidden."""

    name: str
    arm: str
    n: int
    mean: float
    std: float
    min: float
    max: float
    ci_low: float | None = None
    ci_high: float | None = None
    values: list[float] = Field(default_factory=list)


class Comparison(MRAModel):
    """A paired treatment-vs-baseline comparison with its uncertainty."""

    metric: str
    baseline_arm: str
    treatment_arm: str
    baseline: MetricSummary
    treatment: MetricSummary
    effect: float
    relative_effect: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    n_seeds: int = 0
    note: str = ""


class Result(Entity):
    """The Analyst's aggregate over a spec's runs."""

    id: str = Field(default_factory=lambda: new_id("result"))
    spec_id: str
    spec_hash: str
    run_ids: list[str] = Field(default_factory=list)
    summaries: list[MetricSummary] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)
    figures: list[Artifact] = Field(default_factory=list)
    n_seeds: int = 0
    failed_runs: int = 0
    total_cost_usd: float = 0.0
    notes: list[str] = Field(default_factory=list)


class Finding(MRAModel):
    """One Critic objection. The Critic's job is to produce these, not approvals."""

    severity: Literal["blocker", "major", "minor", "note"]
    category: str
    statement: str
    evidence: str = ""
    suggested_fix: str = ""


class CritiqueReport(Entity):
    """The Critic's verdict on a phase artifact, and the gate score it implies."""

    id: str = Field(default_factory=lambda: new_id("crit"))
    phase: Phase
    target_id: str
    findings: list[Finding] = Field(default_factory=list)
    score: Score = 0.0
    passed: bool = False
    summary: str = ""

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]


class Verdict(Entity):
    """supported | refuted | inconclusive -- decided by the rule, not by taste."""

    id: str = Field(default_factory=lambda: new_id("verdict"))
    hypothesis_id: str
    result_id: str
    status: VerdictStatus
    reasoning: str
    decision_rule_statement: str = ""
    rule_fired: bool = False
    seed_variance_note: str = ""
    threats_to_validity: list[str] = Field(default_factory=list)
    critique_id: str | None = None
    confidence: Score = 0.5


class FollowUp(MRAModel):
    """A proposed next experiment, ranked by information gain per dollar."""

    title: str
    rationale: str
    kind: Literal["ablation", "control", "scale_up", "replication", "new_baseline"] = "ablation"
    expected_information_gain: Score = 0.5
    estimated_cost_usd: float = 1.0
    hypothesis_id: str | None = None

    @property
    def gain_per_dollar(self) -> float:
        return self.expected_information_gain / max(self.estimated_cost_usd, 0.01)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Citation(MRAModel):
    """What the Writer is allowed to lean on: a KB source or a run id."""

    key: str
    label: str
    kind: Literal["paper", "run", "repo", "note"] = "paper"
    locator: str | None = None
    url: str | None = None


class ReportSection(MRAModel):
    title: str
    body: str
    level: int = 2
    citations: list[Citation] = Field(default_factory=list)
    figures: list[Artifact] = Field(default_factory=list)
    order: int = 0


class Report(Entity):
    id: str = Field(default_factory=lambda: new_id("report"))
    brief_id: str
    title: str
    abstract: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    path: str | None = None


__all__ = [
    "PHASE_ORDER",
    "Artifact",
    "Author",
    "Citation",
    "Claim",
    "ClaimRelation",
    "CodeRepo",
    "Comparison",
    "CritiqueReport",
    "DatasetRef",
    "DecisionRule",
    "EnvSpec",
    "Entity",
    "Evidence",
    "EvidenceKind",
    "ExperimentSpec",
    "FailureSignature",
    "FalsifiableClaim",
    "Finding",
    "FollowUp",
    "GraphEdge",
    "Hypothesis",
    "Idea",
    "LicenseCategory",
    "LicenseInfo",
    "MRAModel",
    "Metric",
    "MetricDef",
    "MetricReport",
    "MetricSummary",
    "ModelTier",
    "Note",
    "Paper",
    "PaperSection",
    "Passage",
    "Phase",
    "Provenance",
    "Question",
    "RankedPaper",
    "Recipe",
    "RecipeStep",
    "RepoMap",
    "Report",
    "ReportSection",
    "ResearchBrief",
    "Result",
    "RunRecord",
    "RunStatus",
    "Scale",
    "Score",
    "ScreeningDecision",
    "ScreeningDecisionKind",
    "SearchHit",
    "SuccessCriterion",
    "Variable",
    "Venue",
    "Verdict",
    "VerdictStatus",
    "new_id",
    "utcnow",
]
