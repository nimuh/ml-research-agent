#!/usr/bin/env python3
"""Structural checks over a research vault.

Everything here is a check a model would perform unreliably: counting links
across two hundred files, comparing a hash against the bytes it covers,
noticing that one number out of forty has no quote behind it. Those are exactly
the checks worth automating, and the ones a reviewer skims past.

The judgment calls -- does this quote actually support this sentence, are these
three concept notes the same idea -- are deliberately absent. They belong to the
`lint-vault` skill, which reads this output and then goes and looks.

Stdlib only, so it runs with `python3 lint_vault.py <vault>` and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Frontmatter every note carries, per `_meta/schema.md`.
REQUIRED_FIELDS = ("type", "title", "added", "tags")

# Two kinds of file are held to a different contract, because they are not notes
# somebody writes -- they are records something produced, and the note contract
# asks them for fields that would be redundant or invented.
#
# Dossiers in `raw/` are acquisition records: written once by the survey, never
# edited, carrying what the retrieval knew rather than what the vault decided.
# A run is identified by its `id`, `arm` and `seed`; a `title` on it would be a
# restatement of those, and requiring one teaches people to write filler.
BY_TYPE: dict[str, tuple[str, ...]] = {
    "dossier": ("type", "key", "read"),
    "run": ("type", "id", "arm", "seed", "status", "added", "tags"),
}

# Types whose notes must carry a `## Evidence` section holding verbatim quotes.
NEEDS_EVIDENCE = ("source",)

# Directories that are derived output, not notes. Linting them for orphanhood
# reports noise: a report nothing links to is normal.
DERIVED_DIRS = ("reports", "_meta", ".obsidian")

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
# A bare number that carries a claim: percentages, decimals, and figures with
# units. Deliberately not every integer -- years, seeds and section numbers are
# not results, and flagging them would bury the real finding.
CLAIMY_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+|\d{1,3}(?:\.\d+)?\s*%)(?![\w])")
QUOTE_LINE = re.compile(r"^\s*>", re.MULTILINE)

SEVERITIES = ("blocker", "major", "minor")


@dataclass
class Finding:
    severity: str
    check: str
    path: str
    message: str
    fix: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "check": self.check,
            "path": self.path,
            "message": self.message,
            "fix": self.fix,
        }


@dataclass
class Note:
    path: Path
    rel: str
    stem: str
    frontmatter: dict[str, str]
    body: str
    links: set[str] = field(default_factory=set)

    @property
    def type(self) -> str:
        return self.frontmatter.get("type", "")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(raw frontmatter, body)``; empty frontmatter if there is none."""
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[3:end], text[end + 4 :]


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML front matter from the body.

    A deliberately shallow parser: it reads top-level `key: value` pairs and
    leaves the values as raw strings. The checks below only ever ask whether a
    key is present and whether its text looks a certain way, and depending on a
    YAML library would mean this script could not be run with bare python3 --
    which is the property that makes it usable from a skill.
    """
    raw, body = _split_frontmatter(text)
    if not raw:
        return {}, body

    data: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip() or line.startswith("#") or line.startswith((" ", "\t", "-")):
            continue
        key, sep, value = line.partition(":")
        if sep:
            data[key.strip()] = value.strip()
    return data, body


def load_notes(vault: Path) -> list[Note]:
    notes: list[Note] = []
    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault).as_posix()
        if rel.startswith(".obsidian/"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = parse_frontmatter(text)
        note = Note(path=path, rel=rel, stem=path.stem, frontmatter=frontmatter, body=body)
        note.links = {m.group(1).strip() for m in WIKILINK.finditer(body)}
        notes.append(note)
    return notes


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_frontmatter(notes: list[Note]) -> list[Finding]:
    out: list[Finding] = []
    for note in notes:
        if note.rel.startswith(".obsidian/") or note.rel.startswith("_meta/"):
            continue
        kind = "dossier" if note.rel.startswith("raw/") else note.type
        required = BY_TYPE.get(kind, REQUIRED_FIELDS)
        missing = [f for f in required if f not in note.frontmatter]
        if not note.frontmatter:
            out.append(
                Finding(
                    "major",
                    "frontmatter",
                    note.rel,
                    "no YAML front matter",
                    "add the block `_meta/schema.md` specifies for this note type",
                )
            )
        elif missing:
            out.append(
                Finding(
                    "major",
                    "frontmatter",
                    note.rel,
                    f"missing required field(s): {', '.join(missing)}",
                    "see `_meta/schema.md`",
                )
            )
    return out


def check_links(notes: list[Note]) -> list[Finding]:
    """Resolve every `[[link]]` the way Obsidian's shortest-path setting does."""
    by_stem: dict[str, list[str]] = {}
    by_rel: set[str] = set()
    for note in notes:
        by_stem.setdefault(note.stem, []).append(note.rel)
        by_rel.add(note.rel)
        by_rel.add(note.rel.removesuffix(".md"))

    out: list[Finding] = []
    for note in notes:
        for link in sorted(note.links):
            target = link.removesuffix(".md")
            if target in by_stem or target in by_rel or Path(target).name in by_stem:
                continue
            out.append(
                Finding(
                    "major",
                    "broken-link",
                    note.rel,
                    f"[[{link}]] resolves to nothing",
                    "create the note, or remove the link -- a link to a note that "
                    "does not exist is a false statement about the vault",
                )
            )
    return out


def check_orphans(notes: list[Note]) -> list[Finding]:
    linked_to: set[str] = set()
    for note in notes:
        for link in note.links:
            linked_to.add(Path(link.removesuffix(".md")).name)

    out: list[Finding] = []
    for note in notes:
        if note.rel.startswith(DERIVED_DIRS) or note.stem in ("README", "TODO", "log", "map"):
            continue
        # A run record legitimately links nothing -- it is a measurement, not an
        # argument. Whether anything points *at* it is checked from the other
        # side, by `check_runs`, which is the direction that actually matters:
        # a result citing a missing run is a defect, an uncited run is not.
        if note.type == "run":
            continue
        if note.stem in linked_to or note.links:
            continue
        out.append(
            Finding(
                "minor",
                "orphan",
                note.rel,
                "nothing links to it and it links nothing",
                "link the concepts it touches; an island in the graph is usually "
                "a note written without being read",
            )
        )
    return out


def check_evidence(notes: list[Note]) -> list[Finding]:
    """Sources need quotes, and their numbers need to appear inside them."""
    out: list[Finding] = []
    for note in notes:
        if note.type not in NEEDS_EVIDENCE:
            continue
        if note.frontmatter.get("read", "").strip() == "unreachable":
            continue

        head, sep, evidence = note.body.partition("## Evidence")
        if not sep:
            out.append(
                Finding(
                    "blocker",
                    "no-evidence",
                    note.rel,
                    "source note has no `## Evidence` section",
                    "paste the dossier's quotes with their locators, or delete the "
                    "assertions that rest on them",
                )
            )
            continue
        if not QUOTE_LINE.search(evidence):
            out.append(
                Finding(
                    "blocker",
                    "no-evidence",
                    note.rel,
                    "`## Evidence` contains no quoted passage",
                    "quotes are blockquotes copied verbatim from `raw/`",
                )
            )
            continue

        quoted = " ".join(
            line.lstrip("> ").strip() for line in evidence.splitlines() if line.lstrip().startswith(">")
        )
        for match in CLAIMY_NUMBER.finditer(head):
            number = match.group(1).replace(" ", "")
            if number.rstrip("%") in quoted.replace(" ", ""):
                continue
            out.append(
                Finding(
                    "blocker",
                    "unsupported-number",
                    note.rel,
                    f"`{match.group(1)}` appears in the note but in no quote",
                    "find the quote in the dossier, or delete the figure",
                )
            )
    return out


def check_claims(notes: list[Note]) -> list[Finding]:
    out: list[Finding] = []
    for note in notes:
        if note.type != "claim":
            continue
        supported = note.frontmatter.get("supported_by", "[]").strip()
        contradicted = note.frontmatter.get("contradicted_by", "[]").strip()
        empty = supported in ("", "[]") and contradicted in ("", "[]")
        if empty:
            out.append(
                Finding(
                    "major",
                    "unsupported-claim",
                    note.rel,
                    "claim cites neither support nor contradiction",
                    "link the source notes, or delete the claim",
                )
            )
        status = note.frontmatter.get("status", "").strip()
        if status == "settled" and contradicted not in ("", "[]"):
            out.append(
                Finding(
                    "major",
                    "resolved-contest",
                    note.rel,
                    "marked settled while still listing contradicting sources",
                    "a disagreement is a finding -- set status back to contested, or "
                    "explain in the body why the contradiction does not stand",
                )
            )
    return out


#: Front-matter keys a spec's hash covers, in the order `spec_hash.py` uses.
HASHED_KEYS = (
    "id",
    "hypothesis",
    "arms",
    "baseline",
    "dataset",
    "metrics",
    "controls",
    "seeds",
    "scale",
)


def spec_frontmatter(raw: str) -> dict[str, str]:
    """Parse a spec's front matter the way ``spec_hash.py`` does.

    Deliberately *not* :func:`parse_frontmatter`, which drops indented lines --
    that is fine for asking whether a key exists, and wrong for hashing, because
    `decision_rule:` is entirely made of indented lines. Using the loose parser
    here made the linter compute a different digest from the one `spec_hash.py`
    wrote, so every honest spec was reported as tampered. A check that fires on
    correct input is worse than no check: it teaches everyone to ignore the one
    finding that matters most.
    """
    data: dict[str, str] = {}
    key: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t", "-")) and key is not None:
            data[key] = f"{data[key]}\n{line.rstrip()}"
            continue
        name, sep, value = line.partition(":")
        if sep:
            key = name.strip()
            data[key] = value.strip()
    return data


def spec_payload(frontmatter: dict[str, str], body: str) -> str:
    """The bytes a spec's hash covers: its decision-relevant fields.

    Byte-identical to ``design-experiment/scripts/spec_hash.py:payload``. The
    two are separate files because an installed skill has to stand alone;
    ``tests/test_scripts.py`` hashes one fixture through both and compares the
    whole payload, which is what catches the day they drift.
    """
    parts = [f"{k}={_normalise(frontmatter.get(k, ''))}" for k in HASHED_KEYS]

    rule = frontmatter.get("decision_rule", "")
    marker = body.find("## Decision rule")
    if marker != -1:
        end = body.find("\n## ", marker + 1)
        rule = body[marker : end if end != -1 else len(body)]
    parts.append(f"decision_rule={_normalise(rule)}")
    return "\n".join(parts)


def _normalise(value: str) -> str:
    """Collapse whitespace, so re-wrapping a line is not a design change."""
    return " ".join(value.split())


def check_preregistration(vault: Path, notes: list[Note]) -> list[Finding]:
    """The check this whole script exists for.

    A decision rule edited after the numbers arrive turns a falsifiable
    experiment into a story about the data. Nothing about the file looks
    different afterwards -- the hash is the only witness.
    """
    out: list[Finding] = []
    for note in notes:
        if note.type != "spec":
            continue
        recorded = note.frontmatter.get("spec_hash", "").strip().strip("\"'")
        if not recorded:
            out.append(
                Finding(
                    "blocker",
                    "no-spec-hash",
                    note.rel,
                    "pre-registered spec carries no spec_hash",
                    "run `spec_hash.py <spec> --write`; without it, a later edit to "
                    "the decision rule is undetectable",
                )
            )
            continue
        raw, _ = _split_frontmatter(note.path.read_text(encoding="utf-8", errors="replace"))
        actual = hashlib.sha256(
            spec_payload(spec_frontmatter(raw), note.body).encode("utf-8")
        ).hexdigest()[:16]
        if actual != recorded:
            out.append(
                Finding(
                    "blocker",
                    "tampered-preregistration",
                    note.rel,
                    f"spec_hash is {recorded} but the spec now hashes to {actual}",
                    "the spec changed after pre-registration. Restore it from git, or "
                    "re-register as a NEW experiment id -- do not rewrite the hash",
                )
            )
    return out


def check_runs(vault: Path, notes: list[Note]) -> list[Finding]:
    """Every run a results file leans on has to exist."""
    runs = {n.frontmatter.get("id", "").strip().strip("\"'") for n in notes if n.type == "run"}
    run_stems = {n.stem for n in notes if n.type == "run"}

    out: list[Finding] = []
    for note in notes:
        if note.type != "result":
            continue
        for link in sorted(note.links):
            name = Path(link.removesuffix(".md")).name
            if name in run_stems or link in runs:
                continue
            if "seed" not in name:
                continue
            out.append(
                Finding(
                    "blocker",
                    "dangling-run",
                    note.rel,
                    f"cites run `{link}`, which has no record",
                    "a result citing a run nobody can find is unverifiable",
                )
            )
    return out


def check_dossier_coverage(vault: Path, notes: list[Note]) -> list[Finding]:
    """Every dossier is either a source note or a recorded exclusion."""
    raw = vault / "raw"
    if not raw.is_dir():
        return []
    screening = vault / "_meta" / "screening.md"
    screened = (
        screening.read_text(encoding="utf-8", errors="replace") if screening.exists() else ""
    )
    source_raws = {n.frontmatter.get("raw", "").strip().strip("\"'") for n in notes}
    source_stems = {n.stem for n in notes if n.type == "source"}

    out: list[Finding] = []
    for path in sorted(raw.glob("*.md")):
        if path.stem.startswith("_"):
            continue
        rel = path.relative_to(vault).as_posix()
        if rel in source_raws or path.stem in source_stems or path.stem in screened:
            continue
        out.append(
            Finding(
                "major",
                "unscreened-dossier",
                rel,
                "read, but neither noted nor recorded as excluded",
                "write the source note, or record the exclusion and its reason in "
                "`_meta/screening.md`",
            )
        )
    return out


def check_read_state(vault: Path, notes: list[Note]) -> list[Finding]:
    """A note may not claim to have read more than its dossier did."""
    rank = {"unreachable": 0, "abstract-only": 1, "partial": 2, "full": 3}
    out: list[Finding] = []
    for note in notes:
        if note.type != "source":
            continue
        claimed = note.frontmatter.get("read", "").strip().strip("\"'")
        raw_rel = note.frontmatter.get("raw", "").strip().strip("\"'")
        if not claimed or not raw_rel:
            continue
        dossier = vault / raw_rel
        if not dossier.exists():
            out.append(
                Finding(
                    "major",
                    "missing-dossier",
                    note.rel,
                    f"`raw:` points at {raw_rel}, which does not exist",
                    "the dossier is the note's only evidence of having been read",
                )
            )
            continue
        source_fm, _ = parse_frontmatter(dossier.read_text(encoding="utf-8", errors="replace"))
        actual = source_fm.get("read", "").strip().strip("\"'")
        if actual and rank.get(claimed, 3) > rank.get(actual, 0):
            out.append(
                Finding(
                    "blocker",
                    "upgraded-reading",
                    note.rel,
                    f"claims `read: {claimed}` but its dossier says `{actual}`",
                    "a note may never claim more than the dossier read",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def run_checks(vault: Path) -> list[Finding]:
    notes = load_notes(vault)
    findings = [
        *check_frontmatter(notes),
        *check_links(notes),
        *check_evidence(notes),
        *check_claims(notes),
        *check_preregistration(vault, notes),
        *check_runs(vault, notes),
        *check_dossier_coverage(vault, notes),
        *check_read_state(vault, notes),
        *check_orphans(notes),
    ]
    findings.sort(key=lambda f: (SEVERITIES.index(f.severity), f.check, f.path))
    return findings


def render_text(findings: list[Finding], vault: Path) -> str:
    if not findings:
        return f"{vault}: clean -- no structural problems found."

    lines = [f"{vault}", ""]
    counts = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITIES}
    lines.append(
        "  ".join(f"{counts[s]} {s}" + ("s" if counts[s] != 1 else "") for s in SEVERITIES)
    )
    lines.append("")
    current = ""
    for finding in findings:
        if finding.severity != current:
            current = finding.severity
            lines.append(f"── {current.upper()} " + "─" * (60 - len(current)))
        lines.append(f"  {finding.path}")
        lines.append(f"    {finding.check}: {finding.message}")
        if finding.fix:
            lines.append(f"    → {finding.fix}")
    return "\n".join(lines)


def render_todo(findings: list[Finding]) -> str:
    lines = [
        "---",
        "type: concept",
        "title: TODO",
        "tags: [moc]",
        "---",
        "",
        "# TODO",
        "",
        "*Written by `lint-vault`. Regenerated on each run — check items off by",
        "fixing them in the vault, not by editing this file.*",
        "",
    ]
    if not findings:
        lines.append("Nothing outstanding. The vault passes every structural check.")
        return "\n".join(lines) + "\n"

    for severity in SEVERITIES:
        group = [f for f in findings if f.severity == severity]
        if not group:
            continue
        lines.append(f"## {severity.title()}")
        lines.append("")
        for finding in group:
            lines.append(f"- [ ] `{finding.path}` — {finding.message}")
            if finding.fix:
                lines.append(f"      {finding.fix}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vault", type=Path, help="path to the vault root")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--write-todo", action="store_true", help="write TODO.md in the vault")
    parser.add_argument(
        "--fail-on",
        choices=(*SEVERITIES, "never"),
        default="blocker",
        help="exit non-zero when a finding at this severity or worse exists",
    )
    args = parser.parse_args()

    if not args.vault.is_dir():
        print(f"not a directory: {args.vault}", file=sys.stderr)
        return 2

    findings = run_checks(args.vault)

    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        print(render_text(findings, args.vault))

    if args.write_todo:
        (args.vault / "TODO.md").write_text(render_todo(findings), encoding="utf-8")

    if args.fail_on == "never":
        return 0
    ceiling = SEVERITIES.index(args.fail_on)
    return 1 if any(SEVERITIES.index(f.severity) <= ceiling for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
