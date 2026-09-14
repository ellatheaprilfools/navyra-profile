import argparse
import json
import sys

from .analyser import analyse
from .fingerprint import Fingerprinter


def _read_prompts(path, field):
    """Read JSONL `path` and extract the value under `field` from each line.

    Returns a list of prompt strings. Raises `ValueError` for malformed
    JSON or if the expected `field` is missing on any line.
    """
    prompts = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: not valid JSON ({e.msg})") from e

            if field not in obj:
                raise ValueError(
                    f"{path}:{line_no}: missing field '{field}' "
                    f"(found: {list(obj.keys())})"
                )
            prompts.append(obj[field])

    if not prompts:
        raise ValueError(f"{path}: no prompts found — is the file empty?")

    return prompts


def _cmd_analyse(args):
    """Handler for the `analyse` subcommand.

    Reads prompts from the provided file, runs `analyse()` and prints a
    text summary. If `--html` is given, attempts to write an HTML report.
    """
    prompts = _read_prompts(args.file, args.field)
    report = analyse(prompts, radius=args.radius)
    print(report.summary())

    if args.html:
        from . import report as report_mod
        report_mod.html_report(report, args.html)
        print(f"\nwrote HTML report to {args.html}")

    if args.pdf:
        from . import report as report_mod
        report_mod.pdf_report(report, args.pdf)
        print(f"wrote PDF report to {args.pdf}")

def _cmd_validate(args: argparse.Namespace) -> None:
    from .validation import run_sweep
    shares = [float(s) for s in args.shares.split(",")]
    result = run_sweep(template_shares=shares, n=args.n, seed=args.seed)
    print(result.summary())

def _cmd_synth(args: argparse.Namespace) -> None:
    """Handler for the `synth` subcommand.

    Generates synthetic traffic using `make_traffic()` and writes JSONL to
    the requested output path.
    """
    from .synth import make_traffic
    result = make_traffic(
        n=args.n,
        template_share=args.template_share,
        n_templates=args.n_templates,
        paraphrase_strength=args.paraphrase_strength,
        seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        for p in result.prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"wrote {len(result.prompts):,} synthetic prompts to {args.output}")

def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level `argparse.ArgumentParser` for
    the CLI.
    """
    parser = argparse.ArgumentParser(prog="navyra-profile")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyse = sub.add_parser("analyse")
    p_analyse.add_argument("--pdf", default=None, metavar="PATH")
    p_analyse.add_argument("file", help="path to a JSONL file of prompts")
    p_analyse.add_argument("--field", default="prompt")
    p_analyse.add_argument("--radius", type=int, default=6)
    p_analyse.add_argument("--html", default=None, metavar="PATH")   
    p_analyse.set_defaults(func=_cmd_analyse)                        

    p_synth = sub.add_parser("synth")
    p_synth.add_argument("--n", type=int, required=True)
    p_synth.add_argument("--template-share", type=float, required=True)
    p_synth.add_argument("--n-templates", type=int, default=25)
    p_synth.add_argument("--paraphrase-strength", type=float, default=0.3)
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.add_argument("-o", "--output", required=True)
    p_synth.set_defaults(func=_cmd_synth)    

    p_validate = sub.add_parser("validate", help="run the accuracy sweep study")
    p_validate.add_argument("--n", type=int, default=2000)
    p_validate.add_argument("--shares", type=str, default="0.1,0.2,0.4,0.6,0.8")
    p_validate.add_argument("--seed", type=int, default=0)
    p_validate.set_defaults(func=_cmd_validate)                         

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    
if __name__ == "__main__":
    main()
