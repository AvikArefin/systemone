# 02 — What is a Jev-style model?

## The idea

A normal language model reads text and **writes more text**. That is a poor fit
for software that needs a decision: you have to parse the output, hope it is
well-formed, and you cannot get a calibrated probability out of it.

A **System One decision model** — the category TypeSafe AI's **Jev** defined —
does something narrower and more useful:

```
input:   a state  +  a typed question
output:  a probability distribution over the allowed answers
```

There is no text generation. The output space is the question's own candidates,
so a malformed answer is structurally impossible.

Three question types are used throughout this project:

| Type | Question | Output |
|---|---|---|
| `noul` | "Is X true of this state?" | one probability |
| `choice` | "Which of these N options applies?" | distribution over N options |
| `score` | "Which level on this ordered scale?" | distribution over levels, plus expected value |

"System One" is a reference to dual-process psychology: fast, automatic
judgement rather than slow, deliberative reasoning. One forward pass, no chain
of thought.

## Why not just use an LLM?

Because of where this runs. The target is a microcontroller with 8 MB of PSRAM.
That dictates the entire design:

| | a 7B LLM | TinyJev |
|---|---|---|
| Parameters | 7,000M | 1M |
| Weights in memory | ~14 GB fp16 | ~4 MB fp32, ~1 MB int8 |
| Latency | hundreds of ms | single-digit ms |
| Output | free text to parse | bounded distribution |
| Can it hallucinate an invalid answer? | yes | no |

The trade is capability. TinyJev cannot reason about arbitrary instructions the
way a large model can. It is very good at familiar decision shapes and much worse
at genuinely new ones — a limitation the small models in this space all share
(see [Experiments](11-experiments.md)).

## Where this project sits

The **S1MB** benchmark (System One Mosaic Benchmark, by Yuichi Tateno) exists to
compare these models: 137 typed benchmarks over 106 subsets, 26,269 judgments.
The published roster includes:

- **Jev 1.13** (TypeSafe AI) — the strongest, and the one this category is named
  after. Parameter count is not published.
- **OpenJev / Open-Jev-9B / 27B** — open reproductions at LLM scale.
- **bekko-system-one** — 17M / 68M / 400M open encoder models, and the source of
  the public training corpus this project uses.
- **Laya** — Convai's 421M open-weight decision models.
- **This project**, `TinyJev` — 1M parameters, aimed at microcontrollers.

Published Task Avg scores (baseline-adjusted, higher is better):

```
Jev 1.13                59.59
bekko-system-one-400m   50.60
bekko-system-one-68m    40.46
bekko-system-one-17m    27.57
laya-typed-decisions    15.00
laya                    13.36
```

That table is the honest context for anything we report. It is also why "beat
Laya" is a low bar and "beat Jev" is a research programme — read
[Metrics](10-metrics.md) before comparing any number of ours to it.

**This project has no entry in that table, and it should not yet.** The current
checkpoint measures at roughly zero adjusted skill, and the raw accuracy figures
quoted elsewhere were mostly measuring the benchmark's label marginals. The
encoding bug that caused most of that is fixed; a comparable number requires a
retrain. See [Experiments](11-experiments.md).

## What "Jev-style" means architecturally

Not a specific network. It is a **shape of computation**:

1. One shared state.
2. One or more typed questions over it.
3. All candidates for a question scored **in parallel in a single forward pass**,
   not one by one.
4. A softmax **within each question's candidate set**, so answers compete.
5. Soft targets supported, because real annotations are distributions.
6. Everything computed once per state and reused across its questions.

TinyJev implements exactly these six properties. The mechanics are in
[Architecture](03-architecture.md); the batch format that makes property 3 cheap
is in [Data pipeline](05-data-pipeline.md).

Next: [Architecture →](03-architecture.md)
