---
type: source
key: arxiv:2312.00752
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
authors: [Albert Gu, Tri Dao]
year: 2023
venue: COLM 2024
url: https://arxiv.org/abs/2312.00752
code: [https://github.com/state-spaces/mamba]
citations: 4200          # omit if no page stated it
read: full               # full | partial | abstract-only | unreachable
confidence: 0.8          # confidence in this reading, not in the paper
raw: raw/arxiv-2312.00752.md
added: 2026-08-14
tags: [source, ssm, long-context]
---

# {{TITLE}}

{{Authors}} · {{Year}} · {{Venue}} · [paper]({{url}}) · [[{{raw dossier}}|dossier]]

## Summary

{{Three to five sentences: what they did and what they found. Every number here
appears in a quote below, tagged (E1), (E2).}}

## Method

{{As the authors describe it — links every method node it touches: [[Mamba]],
[[selective scan]]. Not your reconstruction of how it probably works.}}

## Setup

Datasets: [[The Pile]], [[Genomic Benchmarks]]
Baselines: [[Transformer++]]
Compute: {{what was actually used, if stated (E4) — "not stated" is an answer}}

## Results

| Metric | Setting | This work | Baseline | Evidence |
|---|---|---|---|---|
| [[bits-per-byte]] | 8k context | 1.02 | 1.09 | E1 |

## Limitations

- {{stated by the authors (E2) — not your critique, which belongs in red-team}}

## Relevance

{{What this does for *this brief*. "Barely anything, but it is the baseline
everyone compares against" is a real and useful answer.}}

Bears on: [[SSMs beat transformers beyond 32k context]]

## Evidence

Copied verbatim from the dossier. Nothing above is asserted without a quote here.

**E1**
> Mamba reaches 1.02 bits-per-byte at 8k context, against 1.09 for the
> transformer baseline of equal parameter count.
— §4.2 Results, p.7 · https://arxiv.org/abs/2312.00752

**E2**
> We do not evaluate on sequences beyond one million tokens.
— §6 Limitations, p.11 · https://arxiv.org/abs/2312.00752
