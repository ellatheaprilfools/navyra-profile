# navyra-profile

Before you accelerate anything, you need to know what's actually there
`navyra-profile` measures semantic repetition in LLM prompt traffic. Given a file of prompts, it reports the proportion that are exact duplicates, semantically similar paraphrases, and near-duplicates differing only in specific details (numbers, dates, names).

## The Problem

You already know your inference bill is too high. What you don't know is why.

Some of it is exact repeats, some of it is the same question, asked differently, some of it are near repeats with the same shape, different number, different date.

navyra-profile tells the three apart. Point it at a file of prompts. Get back a report you can act on.


## Installation

```bash
pip install navyra-profile[full]
```

Prompt data is not transmitted anywhere. The sentence-transformer model (`all-MiniLM-L6-v2`) is downloaded from Hugging Face on first run and cached locally; subsequent runs make no network requests.


## Usage

```bash
navyra-profile analyse examples/sample_traffic.jsonl
```

```
prompts analysed          : 400
T1 exact repeats          : 24.5%  (already free via prefix caching)
T2 same-bucket semantic   : 37.1%  (engine-addressable today; upper-bound proxy)
T3 cross-bucket semantic  : 0.0%   (future-addressable)
T4 near-duplicates flagged: 5      (unsafe for answer-replay caches)
```

To generate a full HTML report with charts and near-duplicate examples:

```bash
navyra-profile analyse examples/sample_traffic.jsonl --html report.html
```

## Report tiers

**T1 — exact repeats.** Byte-for-byte identical text after normalisation (lowercased, whitespace collapsed). Already addressed by prefix caching in most serving stacks. Subtracted from total traffic before T2/T3 rates are calculated.

**T2 — same-bucket semantic.** Different wording, equivalent meaning, comparable token length. Computed as an upper bound from embedding similarity (Hamming distance between 64-bit sign-random-projection fingerprints, within a configurable radius). The engine's actual exploitable hit rate is a separate, internally gated measurement.

**T3 — cross-bucket semantic.** Same detection criteria as T2, different length bucket. Reported separately; not addressed by the current engine.

**T4 — near-duplicates.** Fingerprint distance below a stricter threshold than T2/T3, indicating near-identical text with a differing detail (number, date, name). Reported as a count, not a rate. Caching these as exact repeats risks serving an incorrect answer.

Validation

T1 0.6% · T2 1.4% · T3 0.3% — mean absolute error against known ground truth, swept across five traffic compositions from 10% to 80% templated.

Accuracy is measured against synthetic traffic with known tier labels, generated independently of the analysis pipeline. Cross-bucket (T3) labels are additionally validated against the same fingerprinting mechanism used for detection at generation time.

Full sweep table, per-tier findings, and known limitations: VALIDATION.md.

Full methodology: VALIDATION.md.

## CLI reference
navyra-profile analyse <file.jsonl> [--field prompt] [--radius 6] [--html out.html] [--pdf out.pdf]

--field: JSON key containing the prompt text. --radius: maximum Hamming distance (of 64 bits) for a semantic match. --html/--pdf: write a full report in addition to the terminal summary.

navyra-profile synth --n 10000 --template-share 0.4 -o traffic.jsonl

Generates synthetic traffic with known tier ground truth, used for the validation sweep above.

navyra-profile validate [--n 2000] [--shares 0.1,0.2,0.4,0.6,0.8]

Runs the accuracy sweep against the current build.

## License

Apache-2.0.

## Contributing

Issues and pull requests accepted via the repository's standard GitHub workflow.

## Credits

Navyra Ltd · Company No. 17179469
