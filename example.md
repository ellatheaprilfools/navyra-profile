# Worked examples

## Deciding whether to investigate acceleration

A support team runs a self-hosted model behind their ticketing system. GPU spend has grown with ticket volume. Before evaluating any acceleration approach, they need a number: how much of the traffic is actually redundant.

```bash
navyra-profile synth --n 8000 --template-share 0.5 -o examples/synthetic_support.jsonl
navyra-profile analyse examples/synthetic_support.jsonl --html support_report.html
```

```
prompts analysed          : 8,000
T1 exact repeats          : 7.8%   (already free via prefix caching)
T2 same-bucket semantic   : 13.4%  (engine-addressable today; upper-bound proxy)
T3 cross-bucket semantic  : 4.7%   (future-addressable)
T4 near-duplicates flagged: 5      (unsafe for answer-replay caches)
```

Reading this for a decision, not just as numbers:

T1 at 7.8% is prefix-caching territory. If the serving stack already caches prefixes, this is not new value.

T2 at 13.4% is the number worth acting on. This is reported as an upper bound: the true addressable rate is gated inside a matching engine, not measured directly here.

T4 flagged 5 prompts as near-duplicates: near-identical text differing by a number, date, or name. These are pulled up individually in `support_report.html`, with the differing tokens highlighted. This tier exists because a naive whole-answer cache would serve one customer's ticket status to another.

## Reading the near-duplicate risk box

Open `support_report.html` and scroll to "Near-duplicate risk." A pair from this dataset's known-templated traffic:

```
What's the status of order 93090?
What's the status of order 19406?
```

The order numbers differ. Everything else is identical. A semantic cache matching on overall similarity would likely treat these as the same request. `navyra-profile` flags this pair specifically because the numeric difference is the entire content of the answer.

## Checking the tool's own accuracy before trusting its output

The T2 figure above is a profiler estimate, not a validated measurement of any specific dataset. To see how accurate the profiler's estimates are in general, run the validation sweep against your own build:

```bash
navyra-profile validate
```

This generates synthetic traffic with known true tier rates, runs the profiler against it, and reports the error between what was reported and what was actually true. Full methodology and results: [VALIDATION.md](VALIDATION.md).

This generates synthetic traffic with a known, deliberately labelled composition. `--template-share 0.5` means half the generated traffic is built from repeating templates with filled-in values; the rest is unique. Higher values produce more T1/T2/T3 repetition.