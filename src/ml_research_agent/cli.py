"""Typer/argparse CLI. Thin shell over the orchestrator; no business logic.

Planned commands:
    mra idea "<text>"            Frame an idea into a research brief.
    mra survey <idea-id>         Literature search + KB ingest.
    mra kb search|show|stats     Query the knowledge base.
    mra design <idea-id>         Produce experiment specs from the brief + KB.
    mra run <spec-id>            Execute experiments (local or sandboxed).
    mra report <idea-id>         Emit the final research report.
    mra auto "<text>"            Full loop with human checkpoints.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .config import Config
from .errors import MRAError
from .types import Idea, Phase

app = typer.Typer(
    name="mra",
    help="Agentic ML research: survey the literature, build a KB, run pre-registered experiments.",
    no_args_is_help=True,
    add_completion=False,
)
kb_app = typer.Typer(name="kb", help="Query the knowledge base.", no_args_is_help=True)
app.add_typer(kb_app)

console = Console()

ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Extra YAML config file.")]
ProjectOpt = Annotated[str | None, typer.Option("--project", "-p", help="Project id.")]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Auto-approve all human gates.")]


def _config(config_file: str | None = None, **overrides: Any) -> Config:
    return Config.load(config_file=config_file, overrides=overrides).ensure_paths()


def _director(config: Config, project: str | None) -> Any:
    from .orchestrator.director import ResearchDirector
    from .orchestrator.state import ProjectState

    state = ProjectState.load(config, project) if project else None
    return ResearchDirector(config, state=state)


def _fail(exc: Exception) -> None:
    console.print(f"[bold red]error:[/] {exc}")
    raise typer.Exit(code=1)


@app.command()
def idea(
    text: Annotated[str, typer.Argument(help="The research idea, in one sentence.")],
    domain: Annotated[str | None, typer.Option(help="Field, e.g. 'NLP'.")] = None,
    config_file: ConfigOpt = None,
    yes: YesOpt = False,
) -> None:
    """Frame an idea into a validated research brief."""
    config = _config(config_file, **{"gates.auto_approve": yes or None})
    director = _director(config, None)
    try:
        director.state.set_idea(Idea(text=text, domain=domain))
        outcome = director.run_phase(Phase.FRAME)
    except MRAError as exc:
        _fail(exc)
        return
    brief = director.state.snapshot.brief
    assert brief is not None
    console.print(f"[bold green]project[/] {director.state.project_id}")
    console.print(f"[bold]{brief.title}[/]\n")
    console.print(brief.problem_statement + "\n")
    for claim in brief.claims:
        console.print(f"  • {claim.statement}")
        console.print(f"    [dim]refuted by: {claim.falsifier}[/]")
    for criterion in brief.success_criteria:
        console.print(
            f"  ✓ {criterion.metric} {criterion.comparator} {criterion.threshold}"
            + (f" on {criterion.dataset}" if criterion.dataset else "")
        )
    if not outcome.ok:
        raise typer.Exit(code=1)
    director.close()


@app.command()
def survey(
    project: Annotated[str, typer.Argument(help="Project id from `mra idea`.")],
    config_file: ConfigOpt = None,
    yes: YesOpt = False,
) -> None:
    """Search the literature and build the knowledge base."""
    config = _config(config_file, **{"gates.auto_approve": yes or None})
    director = _director(config, project)
    try:
        for phase in (Phase.SURVEY, Phase.CURATE):
            outcome = director.run_phase(phase)
            console.print(f"[bold]{phase.value}[/]: {outcome.summary}")
            if not outcome.ok:
                raise typer.Exit(code=1)
    except MRAError as exc:
        _fail(exc)
    director.close()


@app.command()
def design(
    project: Annotated[str, typer.Argument()],
    config_file: ConfigOpt = None,
    yes: YesOpt = False,
) -> None:
    """Ground the idea in code, synthesize the literature, and pre-register specs."""
    config = _config(config_file, **{"gates.auto_approve": yes or None})
    director = _director(config, project)
    try:
        for phase in (Phase.GROUND, Phase.SYNTHESIZE, Phase.DESIGN):
            outcome = director.run_phase(phase)
            console.print(f"[bold]{phase.value}[/]: {outcome.summary}")
            if not outcome.ok:
                raise typer.Exit(code=1)
    except MRAError as exc:
        _fail(exc)
    for spec in director.state.snapshot.specs:
        console.print(f"\n[bold]{spec.title}[/] [dim]{spec.spec_hash}[/]")
        console.print(f"  rule: {spec.decision_rule.statement}")
        console.print(f"  arms: {', '.join(spec.arms)} | seeds: {spec.seeds}")
    director.close()


@app.command("run")
def run_cmd(
    project: Annotated[str, typer.Argument()],
    config_file: ConfigOpt = None,
    yes: YesOpt = False,
) -> None:
    """Implement, execute and analyze the pre-registered experiments."""
    config = _config(config_file, **{"gates.auto_approve": yes or None})
    director = _director(config, project)
    try:
        for phase in (Phase.IMPLEMENT, Phase.RUN, Phase.ANALYZE):
            outcome = director.run_phase(phase)
            console.print(f"[bold]{phase.value}[/]: {outcome.summary}")
            if not outcome.ok:
                raise typer.Exit(code=1)
    except MRAError as exc:
        _fail(exc)
    director.close()


@app.command()
def report(
    project: Annotated[str, typer.Argument()],
    config_file: ConfigOpt = None,
    yes: YesOpt = False,
) -> None:
    """Write the research memo from state + KB + results."""
    config = _config(config_file, **{"gates.auto_approve": yes or None})
    director = _director(config, project)
    try:
        outcome = director.run_phase(Phase.REPORT)
    except MRAError as exc:
        _fail(exc)
        return
    console.print(outcome.summary)
    director.close()


@app.command()
def auto(
    text: Annotated[str, typer.Argument(help="The research idea, in one sentence.")],
    until: Annotated[str, typer.Option(help="Stop after this phase.")] = "report",
    config_file: ConfigOpt = None,
    yes: YesOpt = False,
) -> None:
    """Run the whole loop, pausing at the configured human gates."""
    config = _config(config_file, **{"gates.auto_approve": yes or None})
    director = _director(config, None)
    try:
        state = director.run(Idea(text=text), until=Phase(until))
    except MRAError as exc:
        _fail(exc)
        return
    console.print(f"[bold green]project[/] {state.project_id}")
    for line in state.transcript(limit=40):
        console.print(f"[dim]{line}[/]")
    director.close()


@app.command()
def status(project: ProjectOpt = None, config_file: ConfigOpt = None) -> None:
    """Show where a project got to, and what it cost."""
    from .orchestrator.state import ProjectState

    config = _config(config_file)
    if project is None:
        projects = ProjectState.list_projects(config)
        if not projects:
            console.print("[dim]no projects yet[/]")
            return
        for name in projects:
            console.print(name)
        return

    state = ProjectState.load(config, project)
    snap = state.snapshot
    table = Table(title=f"project {project}", show_header=False)
    table.add_row("phase", snap.phase.value)
    table.add_row("completed", ", ".join(p.value for p in snap.completed_phases) or "-")
    table.add_row("brief", snap.brief.title if snap.brief else "-")
    table.add_row("papers", str(len(snap.paper_keys)))
    table.add_row("notes", str(len(snap.note_ids)))
    table.add_row("recipes", str(len(snap.recipes)))
    table.add_row("specs", str(len(snap.specs)))
    table.add_row("runs", f"{sum(1 for r in snap.runs if r.succeeded)}/{len(snap.runs)} ok")
    table.add_row("verdicts", ", ".join(v.status.value for v in snap.verdicts) or "-")
    table.add_row("cost", f"${snap.cost_usd:.2f}")
    console.print(table)
    if snap.truncations:
        console.print("\n[bold]truncations[/]")
        for item in snap.truncations:
            console.print(f"  [yellow]{item.get('message', item)}[/]")
    if snap.todos:
        console.print("\n[bold]todos[/]")
        for todo in snap.todos:
            console.print(f"  • {todo}")


@kb_app.command("search")
def kb_search(
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option("-k", help="Results to return.")] = 8,
    min_score: Annotated[
        float | None,
        typer.Option("--min-score", help="Relevance floor; 0 returns raw nearest neighbours."),
    ] = None,
    config_file: ConfigOpt = None,
) -> None:
    """Query the knowledge base and print cited passages."""
    from .knowledge.search import build_retriever

    config = _config(config_file)
    result = build_retriever(config).search(query, k=k, min_score=min_score)
    if not result.hits:
        console.print("[dim]no hits[/]")
        return
    for hit in result.hits:
        console.print(
            f"\n[bold]{hit.citation or hit.passage.locator}[/] [dim]score={hit.score:.3f}[/]"
        )
        console.print(hit.passage.text[:500].strip())


@kb_app.command("show")
def kb_show(note_id: Annotated[str, typer.Argument()], config_file: ConfigOpt = None) -> None:
    """Print one note as stored on disk."""
    from .knowledge.schema import note_to_markdown
    from .knowledge.store import KnowledgeStore

    config = _config(config_file)
    with KnowledgeStore(config) as store:
        note = store.get_note(note_id)
        if note is None:
            console.print(f"[red]no such note:[/] {note_id}")
            raise typer.Exit(code=1)
        console.print(note_to_markdown(note))


@kb_app.command("stats")
def kb_stats(config_file: ConfigOpt = None) -> None:
    """Counts, coverage and health of the knowledge base."""
    from .knowledge.store import KnowledgeStore

    config = _config(config_file)
    with KnowledgeStore(config) as store:
        table = Table(title="knowledge base", show_header=False)
        for key, value in store.stats().items():
            table.add_row(str(key), str(value))
    console.print(table)


@kb_app.command("health")
def kb_health(
    write_todos: Annotated[
        bool, typer.Option("--write-todos", help="Write TODO notes for gaps.")
    ] = False,
    config_file: ConfigOpt = None,
) -> None:
    """Surface orphan notes, uncited claims, contradictions and coverage gaps."""
    from .knowledge.graph import KnowledgeGraph
    from .knowledge.health import KnowledgeHealth
    from .knowledge.store import KnowledgeStore

    config = _config(config_file)
    with KnowledgeStore(config) as store:
        health = KnowledgeHealth(config, store, KnowledgeGraph(store))
        report_ = health.check()
        console.print(report_.to_markdown())
        if write_todos:
            for path in health.write_todos(report_):
                console.print(f"[dim]wrote {path}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
