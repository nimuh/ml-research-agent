# CLAUDE.md

Guidance for Claude Code working **on this repository**. For working *with* the
skills, read `README.md`; for how they fit together, `docs/DESIGN.md`.

## What this repo is

Eleven Claude Code skills plus four stdlib Python scripts. That is all of it.
There is no package, no dependencies, no build step, and nothing to install
beyond `./install.sh`, which symlinks `skills/` into `~/.claude/skills`.

It used to be a 9,000-line Python system — an LLM client, six literature source
adapters, a vector index, an agent framework, a phase orchestrator. That is gone
on purpose. It was scaffolding for things Claude Code already does, and the
scaffolding was where the failures were. **The strong default when adding
anything here is markdown instructions, not code.**

## Commands

```bash
python3 tests/test_scripts.py      # 40 tests, stdlib, offline, ~0.3s
./install.sh --dry-run             # show what would be linked
./install.sh                       # link into ~/.claude/skills
./install.sh --uninstall           # remove only the links pointing at this repo
```

Skills are discovered when a Claude Code session starts, so a new skill or a
renamed one needs a fresh session before it can be invoked.

## Layout

```
skills/<name>/
  SKILL.md         frontmatter + instructions — this IS the spec
  scripts/         stdlib Python, only where determinism is required
  references/      templates the skill writes into a vault
```

`skills/research-idea/references/schema.md` is the vault contract. It is copied
*into* each vault at `_meta/schema.md` so a vault is self-describing and the
skills stay decoupled — every other skill reads the contract from the vault it
is operating on, not from here. Change the schema in both places or not at all.

## When to write a script instead of an instruction

Four scripts exist. Each replaces something a model does unreliably and
confidently:

| Script | Why it is not prose |
|---|---|
| `spec_hash.py` | A pre-registration is worthless if editing it is undetectable |
| `stats.py` | Reading a mean off three numbers is where a model is least trustworthy |
| `decide.py` | Applying a pre-registered rule must not be a judgment call |
| `lint_vault.py` | Comparing a hash to its bytes; counting links across 200 files |

The test: **is it arithmetic, hashing, or exhaustive counting?** If yes, script
it. If it is extraction, judgment, search or writing, it belongs in a `SKILL.md`
— putting it in Python is the mistake this repo was built out of.

Scripts are stdlib-only so a skill can call them with bare `python3`. That is
why Student's t is implemented by hand in `stats.py` rather than imported from
scipy, and why the frontmatter parsers are hand-rolled rather than using yaml.

## Two duplications that are deliberate

**`spec_payload` exists in both `spec_hash.py` and `lint_vault.py`.** An
installed skill has to stand alone, so they cannot share a module.
`test_the_linter_hashes_a_spec_byte_for_byte_the_same_way` compares the *whole*
payload through both — an earlier version compared only the decision-rule
fragment, the two parsers silently disagreed on every other field, and the
linter reported every honest spec as tampered. If you touch either function,
that test is the one that matters.

**Templates repeat the schema.** `references/*.md` restate frontmatter that
`_meta/schema.md` also specifies. A skill has to be usable without reading the
schema file, so this stays — but the schema wins any disagreement.

## Writing a SKILL.md

The frontmatter `description` is how Claude decides to invoke it. It must say
**when to use it, in the words a user would actually type** — a description that
only describes what the skill is never gets triggered. `tests/test_scripts.py`
enforces a trigger clause and a minimum length.

The body: say *why*, not just what. A capable model will reason its way around
any instruction that looks arbitrary unless the reason is on the page — which is
why nearly every rule in these skills carries the failure it prevents. Each skill
ends with a "getting this wrong" section, and those sections do more work than
the instructions above them.

Keep the register plain. No emoji, no hedging, no restating the task back.

## Invariants

`docs/DESIGN.md` §4 lists them. They are load-bearing, not stylistic — every one
is there because of a specific way this kind of system produces confident, wrong
output. Before changing a skill, check whether §4 needs the same edit. The ones
most easily broken by a well-meaning simplification:

- Retrieval never screens; screening is a separate phase with recorded reasons.
- The decision rule is hashed before any run exists, and a design change means a
  new experiment id — never an edit and a re-hash.
- A clean null is `refuted`, not `inconclusive`.
- Contested claims keep both sides.

## Testing

Every script change needs a test. The most valuable ones are adversarial — a
number with no quote behind it, a note claiming to have read more than its
dossier, a spec edited after registration, a significant effect below the
pre-registered threshold. Those tests are regression tests for the system's
*judgment*, and they are the reason to trust anything it outputs.

A test that only asserts the happy path is close to worthless here: the failure
modes are all cases where the output looks fine.
