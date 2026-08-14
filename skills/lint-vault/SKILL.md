---
name: lint-vault
description: Check a research vault for structural problems — broken wikilinks, orphan notes, missing frontmatter, claims with no support, assertions with no quote behind them, and pre-registered decision rules that were edited after runs existed. Use to audit or health-check a knowledge base or Obsidian vault, or after building or extending one.
argument-hint: "[path to vault]"
user-invocable: true
---

# lint-vault

Check that the vault says only what it can show you, and write the gaps down.

Most of this is mechanical, so a script does it — a model eyeballing two hundred
notes for broken links will miss some and will not miss the same ones twice.
What the script cannot judge, you judge afterwards.

## Steps

### 1. Run the checker

```bash
python3 scripts/lint_vault.py <vault-path>
```

Stdlib only, no install. It reports:

| Check | Why it matters |
|---|---|
| Broken `[[links]]` | A link to a note that does not exist is a false statement about what the vault contains |
| Orphan notes | Nothing links in and it links nothing out — usually a note written without being read |
| Missing frontmatter | Breaks the schema every other skill relies on |
| Sources with no `## Evidence` | The note asserts things with nothing behind them |
| Numbers with no quote | A figure in a note that appears in no quote in that note |
| Claims with no support | `supported_by` and `contradicted_by` both empty |
| Unresolved `read:` state | A source note claiming full text whose dossier says otherwise |
| **Tampered pre-registration** | A `spec.md` whose `spec_hash` no longer matches its own decision rule |
| Dangling run references | `results.md` citing runs that are not in `runs/` |

Add `--json` for machine-readable output, `--write-todo` to have it write
`TODO.md` directly.

**The tampered pre-registration check is the one that matters most.** Everything
else is hygiene. A decision rule edited after the numbers came in turns a
falsifiable experiment into a story about the data, and it is invisible without
the hash — that is the entire reason `design-experiment` writes one.

### 2. Judge what the script cannot

Read the vault yourself for the things no checker catches:

- **Notes whose evidence does not support them.** Pull a sample of five source
  notes. For each, take one number in `## Results` and check the quote in
  `## Evidence` actually states it. A quote that is *near* the claim but does not
  contain it is the most common real defect and no regex finds it.
- **A quote that does not appear in its dossier.** Spot-check against `raw/`. If
  a note's quote is not in the dossier verbatim, something invented it.
- **Concept sprawl** — the same idea under three names, splitting the graph.
- **Contested claims quietly resolved** — a `status: settled` claim that still
  has entries under `contradicted_by`.
- **Staleness.** Sources older than the field's current state of the art, when a
  survey has run since. The vault should say what it does not cover.

### 3. Write it down

Everything found becomes a TODO with a location and a fix, in `TODO.md`:

```markdown
- [ ] `sources/arxiv-2401.00001.md` — 71.2% accuracy in `## Results` appears in
      no quote in `## Evidence`. Re-read `raw/arxiv-2401.00001.md`, or delete
      the figure.
```

A gap with a next action is a task. A gap without one is a complaint, and it
will still be there next month.

### 4. Report

Append to `log.md` and tell the user the counts by severity, plus the single
most important thing to fix. If the vault is clean, say that plainly — a clean
lint is a real result and should not be padded.

## Severity

- **Blocker** — tampered pre-registration, a quote not in its dossier, a number
  with no evidence. The vault is asserting things it cannot support; fix before
  anything downstream reads it.
- **Major** — broken links, claims with no support, missing frontmatter. The
  vault still works; parts of it are unnavigable or unchecked.
- **Minor** — orphans, concept sprawl, staleness. Worth a pass when convenient.

Do not report minors as if they were blockers. A lint that cries wolf gets run
once.
