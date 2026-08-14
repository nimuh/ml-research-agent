---
type: claim
title: "SSMs beat transformers beyond 32k context"
status: contested        # settled | contested | open | refuted
supported_by: ["[[arxiv-2312.00752]]", "[[arxiv-2302.10866]]"]
contradicted_by: ["[[arxiv-2405.11111]]"]
added: 2026-08-14
tags: [claim]
---

# {{Claim, stated as a proposition that could be false}}

## Status: contested

## Supporting

- [[arxiv-2312.00752]] (E1) — lower bits-per-byte at 8k on [[The Pile]], equal
  parameters
- [[arxiv-2302.10866]] (E3) — same direction at 16k, different architecture

## Contradicting

- [[arxiv-2405.11111]] (E2) — no gap at 64k once the transformer baseline is
  tuned for long context

## Why they disagree

{{The most valuable paragraph in the vault. Different datasets? Different
baseline tuning? Different definition of "matched compute"? Often the answer is
a confound, and naming it is the experiment worth running.}}

Bears on: [[brief]] C1
