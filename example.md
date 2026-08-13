# Worked examples

## Deciding whether to investigate acceleration

A support team runs a self-hosted model behind their ticketing system. GPU spend has grown with ticket volume. Before evaluating any acceleration approach, they need a number: how much of the traffic is actually redundant.

```bash
navyra-profile analyse support_prompts.jsonl --html support_report.html
```

```
prompts analysed          : 8,400
T1 exact repeats          : 6.2%   (already free via prefix caching)
T2 same-bucket semantic   : 31.4%  (engine-addressable today; upper-bound proxy)
T3 cross-bucket semantic  : 4.1%   (future-addressable)
T4 near-duplicates flagged: 41     (unsafe for answer-replay caches)
```

Reading this for a decision, not just as numbers:

T1 at 6.2% is prefix-caching territory. If the serving stack already caches prefixes, this is not new value.

T2 at 31.4% is the number worth acting on. Roughly a third of traffic has semantic near-duplicates within the same length range. This is reported as an upper bound: the true addressable rate is gated inside a matching engine, not measured directly here.

T4 flagged 41 prompts as near-duplicates: near-identical text differing by a number, date, or name. These are pulled up individually in `support_report.html`, with the differing tokens highlighted. This tier exists because a naive whole-answer cache would serve one customer's ticket status to another.

## Reading the near-duplicate risk box

Open `support_report.html` and scroll to "Near-duplicate risk." A typical entry:

```
Order status for ticket #48213, opened 3 days ago.
Order status for ticket #48291, opened 3 days ago.
```

The ticket numbers differ. Everything else is identical. A semantic cache matching on overall similarity would likely treat these as the same request. `navyra-profile` flags this pair specifically because the numeric difference is the entire content of the answer.

## Checking the tool's own accuracy before trusting its output

The T2 figure above is a profiler estimate, not a validated measurement of any specific dataset. To see how accurate the profiler's estimates are in general, run the validation sweep against your own build:

```bash
navyra-profile validate
```

This generates synthetic traffic with known true tier rates, runs the profiler against it, and reports the error between what was reported and what was actually true. Full methodology and results: [VALIDATION.md](VALIDATION.md).

## Generating comparable synthetic traffic

To reproduce a report shape similar to the support ticket example above, or to test the profiler against traffic of known composition:

```bash
navyra-profile synth --n 8000 --template-share 0.5 -o synthetic_support.jsonl
navyra-profile analyse synthetic_support.jsonl
```

`--template-share` controls the fraction of generated traffic built from repeating templates. Higher values produce more T1/T2/T3 repetition; lower values produce closer to fully unique traffic.