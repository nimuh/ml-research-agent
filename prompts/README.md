# Prompt assets

Prompts are **data, not code**. They live here as versioned files so they can be
diffed in review, tested against golden cases, and referenced by version from a
`ProjectState` event (`agent:framer@v1`) — none of which is possible for a string
literal buried in an agent module.

## Layout

```
prompts/
  <agent-name>/
    v1.md
    v2.md        # new version, never an edit-in-place of v1
```

`PromptLibrary.get("framer")` resolves to the highest version; pass
`version="v1"` to pin. Versions sort numerically, so `v10` comes after `v2`.

## File format

YAML front matter, then the user-message template as the body:

```markdown
---
system: |
  The system prompt. Multi-line via a YAML block scalar.
description: One line on what this prompt is for.
inputs: [idea_text, constraints]
---

The user-message template, with {placeholders} filled by `Prompt.render(...)`.
```

- `system:` is **required**; a prompt with no system message is a `ConfigError`.
- Every other front-matter key lands in `Prompt.metadata` — use it for
  `description`, `inputs`, `output_model`, and any eval notes.
- The body is required and must not be empty.

## Templating rules

`Prompt.render(**variables)` uses `str.format` semantics:

- `{name}` is substituted; a missing variable raises `ConfigError` naming it
  (rather than shipping a literal `{name}` to the model).
- **Literal braces must be doubled**: write `{{"key": "value"}}` to show a JSON
  example in a prompt body.

## Versioning

Never edit a published version in place — a prompt change that silently alters
behaviour makes every prior run unreproducible. Copy `v1.md` to `v2.md`, edit
there, and let the golden-prompt tests cover both until `v1` is retired.
