# systemone documentation

This folder explains the project from the ground up. Each file does one job. If
you are new to machine learning, read them in order; if you only want one thing,
jump straight to it.

**One-paragraph version:** `TinyJev` is a ~1 million parameter neural network that
answers *typed questions* about a piece of text. You give it a state (a message,
a ticket, a document) and one or more questions — yes/no, pick-one, or pick-a-
number — and it returns probabilities, not generated text. It is built to run on
an ESP32-class microcontroller, so it is deliberately tiny and trained on
decision data rather than general web text.

## Read in this order

| # | File | It answers |
|---|---|---|
| 01 | [Glossary](01-glossary.md) | What do all these words mean? |
| 02 | [What is a Jev-style model](02-what-is-jev.md) | What is Jev, and why is this not an LLM? |
| 03 | [Architecture](03-architecture.md) | What is inside the network, layer by layer? |
| 04 | [Tokenizer and prompts](04-tokenizer-and-prompts.md) | How does text become numbers, and how is a question laid out? |
| 05 | [Data pipeline](05-data-pipeline.md) | How do examples become batches, and why a flat token cache? |
| 06 | [Training](06-training.md) | What is being optimised, and how? |
| 07 | [Inference](07-inference.md) | What happens on one real request? |
| 08 | [Calibration](08-calibration.md) | Are the probabilities believable? |
| 09 | [Datasets and benchmark](09-datasets-and-benchmark.md) | Where the data comes from, and what the benchmark is |
| 10 | [Metrics](10-metrics.md) | Why "accuracy" is not the number the leaderboard prints |
| 11 | [Experiments and results](11-experiments.md) | What we ran, what we measured, what it means |
| 12 | [Code map](12-code-map.md) | Which file does what |
| 13 | [Deployment](13-deployment.md) | Why 1M parameters, and what fits on a chip |
| 14 | [Data browser](14-data-browser.md) | How do I look at the data by hand? |

## Status: retrained, first comparable number

The encoding bug is fixed and the model has been retrained on clean rows. The
first Task Avg this project can honestly quote:

```
Task Avg = 1.04        (laya 13.36, bekko-17m 27.57, jev 59.59)
  noul   2.71
  choice 0.40
  score  0.00
```

That is a low number but a real one. `choice` raw accuracy rose 0.308 → 0.362,
which is the kind the bug damaged and the only kind that improved. `score` is flat
zero across all 21 of its benchmarks, which is the clearest thing to fix next.

Two caveats worth keeping in mind:

- This was 1 epoch on 300 cases/subset. Nothing indicates convergence.
- Old figures in these docs (0.534 raw accuracy and the conclusions drawn from it)
  came from rows that had lost their candidates. They are kept, labelled, in
  [Experiments](11-experiments.md), not deleted.

The encoding, the metric, and the loader are fixed and pinned by 145 tests.

## The 30-second mental model

```
text  ──► tokens ──► one vector per token ──► 4 transformer blocks ──► one score per candidate
                                                                          │
                          softmax over candidates of the same question ◄──┘
                                                                          │
                                          yes/no probability, or pick-one, or numeric score
```

Everything else in this folder is detail on one arrow of that diagram.

## Conventions used in these docs

- **State**: the thing being judged (a message, a ticket, a paragraph).
- **Question**: the typed thing you ask about the state.
- **Candidate**: one possible answer to a `choice` or one level of a `score`.
- **Row**: one (state, question, candidate) triple fed to the network. A single
  `choice` question with 4 options becomes 4 rows that share one state.
- Numbers written like `1,001,472` are exact; `~1M` is rounded.
- Code blocks marked `bash` are commands you can run.
- Anything measured on this machine is labelled with the configuration used.
