#!/usr/bin/env python3
"""Hash an experiment spec over the fields that decide its outcome.

A pre-registered decision rule is only worth something if editing it later is
detectable. Nothing about an edited `spec.md` looks different -- the mtime moves
and that is all -- so the hash is the only witness that the rule someone is now
citing is the rule they committed to before the numbers arrived.

What the hash covers is therefore chosen carefully: the hypothesis, the arms and
baseline, the dataset, the metrics, the controls, the seeds, the scale, and the
decision rule. Not the title, not the prose, not `status` -- those move as the
experiment progresses and hashing them would make every legitimate update look
like tampering, which trains everyone to ignore the check.

Usage:
    spec_hash.py spec.md              # print the hash
    spec_hash.py spec.md --write      # write it into the front matter
    spec_hash.py spec.md --verify     # exit 1 if the recorded hash is stale
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

#: Front-matter keys the hash covers, in a fixed order so the digest is stable.
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

HASH_LENGTH = 16


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter, body)``; empty frontmatter if there is none."""
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[3:end], text[end + 4 :]


def parse_frontmatter(raw: str) -> dict[str, str]:
    """Top-level ``key: value`` pairs, values left as raw strings.

    Shallow by design. The digest is taken over the *text* of each value rather
    than a parsed structure, so a reformatting of the YAML is a different hash --
    which is the conservative direction to be wrong in. It also means this runs
    under bare ``python3`` with no yaml dependency, which is what lets a skill
    call it without an install step.
    """
    data: dict[str, str] = {}
    key: str | None = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t", "-")) and key is not None:
            # A nested block (`decision_rule:` and its fields) belongs to the
            # key that opened it; keep it verbatim so the rule is covered.
            data[key] = f"{data[key]}\n{line.rstrip()}"
            continue
        name, sep, value = line.partition(":")
        if sep:
            key = name.strip()
            data[key] = value.strip()
    return data


def payload(frontmatter: dict[str, str], body: str) -> str:
    """The exact bytes the digest is taken over.

    Kept byte-identical to ``lint-vault/scripts/lint_vault.py:spec_payload``.
    They are duplicated rather than shared because each skill has to stand alone
    once installed, and `tests/test_scripts.py` hashes the same fixture through
    both to catch the day they drift.
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


def compute(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    raw, body = split_frontmatter(text)
    digest = hashlib.sha256(payload(parse_frontmatter(raw), body).encode("utf-8"))
    return digest.hexdigest()[:HASH_LENGTH]


def recorded(path: Path) -> str:
    raw, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    match = re.search(r"^spec_hash:\s*(.+)$", raw, re.MULTILINE)
    return match.group(1).strip().strip("\"'") if match else ""


def write(path: Path, digest: str) -> None:
    text = path.read_text(encoding="utf-8")
    raw, body = split_frontmatter(text)
    if re.search(r"^spec_hash:", raw, re.MULTILINE):
        raw = re.sub(r"^spec_hash:.*$", f"spec_hash: {digest}", raw, count=1, flags=re.MULTILINE)
    else:
        lines = raw.splitlines()
        # After `id:` if there is one, so the hash sits with the identity fields.
        at = next((i for i, line in enumerate(lines) if line.startswith("id:")), len(lines) - 1)
        lines.insert(at + 1, f"spec_hash: {digest}")
        raw = "\n".join(lines)
    path.write_text(f"---{raw}\n---{body}", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="write the hash into the spec")
    group.add_argument("--verify", action="store_true", help="fail if the recorded hash is stale")
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"no such spec: {args.spec}", file=sys.stderr)
        return 2

    digest = compute(args.spec)

    if args.write:
        write(args.spec, digest)
        print(f"{args.spec}: spec_hash {digest}")
        return 0

    if args.verify:
        stored = recorded(args.spec)
        if not stored:
            print(f"{args.spec}: no spec_hash recorded", file=sys.stderr)
            return 1
        if stored != digest:
            print(
                f"{args.spec}: PRE-REGISTRATION CHANGED\n"
                f"  recorded: {stored}\n"
                f"  actual:   {digest}\n"
                "  The spec was edited after it was registered. Restore it, or "
                "register the change as a new experiment id -- do not rewrite the hash.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.spec}: intact ({digest})")
        return 0

    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
