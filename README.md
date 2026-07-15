# navyra-profile — summer project starter kit

Welcome, Ella! This zip contains everything needed to start. Read this
file first, then the project brief PDF, then the code in this order:
`fingerprint.py` → `analyzer.py` → the three stub files.

## What this is

Navyra's acceleration engine (which is NOT in this package, and which you
won't need) speeds up LLM inference when traffic is semantically
repetitive. Before anyone installs the engine, they need an answer to one
question: **"how repetitive is MY traffic?"**

`navyra-profile` is the open-source tool that answers it — honestly,
as a TIERED WATERFALL rather than one flattering number:

  T1  exact repeats        -> already served free by vLLM prefix caching
  T2  same-bucket semantic -> addressable by Navyra's engine today
                              (upper-bound proxy; the trial measures truth)
  T3  cross-bucket semantic-> future-addressable
  T4  near-duplicates      -> the danger class: prompts that differ only
                              in a date/number/name. Answer-replay caches
                              (GPTCache-style) silently serve WRONG
                              answers here; the report flags them.

That honesty is the product's credibility. Your project is to take the
working research code in this zip to a polished, pip-installable,
documented, validated open-source package built around that waterfall.

## What's here

```
navyra_profile/
  fingerprint.py   WORKING — embeddings -> 64-bit semantic fingerprints
  analyzer.py      WORKING — streaming would-hit analysis, clusters, buckets
  report.py        STUB — task 3: HTML/terminal report generation
  synth.py         STUB — task 4: synthetic traffic with known ground truth
  cli.py           STUB — task 2: the command-line interface
examples/
  sample_traffic.jsonl   400 synthetic prompts (~55% templated) to play with
tests/
  test_fingerprint.py    starter tests — extend these substantially
pyproject.toml           package config (already pip-installable in dev mode)
```

## Setup (15 minutes)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[full]"        # includes sentence-transformers, matplotlib
pip install pytest
python -m pytest tests/ -q      # should pass 3/3
```

Then try the working core:

```python
import json
from navyra_profile import Fingerprinter, analyse
prompts = [json.loads(l)["prompt"] for l in open("examples/sample_traffic.jsonl")]
print(analyse(prompts).summary())
```

## The tasks (full details, hours and sequencing in the PDF brief)

1. **Familiarise** — run the above, read the code, write down questions.
2. **Package & CLI** — make `pip install navyra-profile` +
   `navyra-profile analyse traffic.jsonl` real. Spec in `cli.py`.
3. **Report generator** — the one-page HTML waterfall report a GPU owner
   forwards to their boss, including the near-duplicate risk box. Spec in
   `report.py`. This is the flagship deliverable.
4. **Synthetic traffic** — generators with known ground-truth shares of
   each tier. Spec in `synth.py`.
5. **Validation harness** — prove each reported tier tracks its ground
   truth across a sweep; quantify per-tier accuracy; grow the test suite.
   (Stretch goal if time allows: scale `analyse()` past 50k prompts —
   see its docstring.)
6. **Documentation** — README for the public repo, worked examples,
   docstrings throughout.

## Ways of working

- Weekly 30–60 min call with Jim (4-week project, ~20 hrs/week);
  questions between calls by email — ask early, ask often.
- Git from day one; small commits with clear messages.
- Definition of done for the project: a stranger can
  `pip install navyra-profile`, run one command on their own JSONL, and
  get a report they trust — because the validation study says why they
  should.

## One boundary

This package is measurement only, and it will be open-sourced. Navyra's
engine (the acceleration technology) is separate, confidential, and out
of scope — if anything you build seems to need engine internals, stop
and ask Jim instead. Everything in this zip is fair game.
