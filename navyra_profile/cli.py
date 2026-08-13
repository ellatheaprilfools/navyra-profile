"""
navyra_profile.cli — command-line interface.  [TO BUILD]

PROJECT TASK 2 (part of packaging). Target UX:

  navyra-profile analyse traffic.jsonl --field prompt --radius 6
  navyra-profile analyse traffic.jsonl --html report.html
  navyra-profile synth --n 10000 --template-share 0.4 -o synth.jsonl

Use argparse or typer. JSONL in, report out. Errors should be friendly.
"""

import argparse
import json
import sys

from .analyser import analyse
from .fingerprint import Fingerprinter


def _read_prompts(path, field):
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
    return prompts


def _cmd_analyse(args):
    prompts = _read_prompts(args.file, args.field)
    report = analyse(prompts, radius=args.radius)
    print(report.summary())

    if args.html:
        from . import report as report_mod
        try:
            report_mod.html_report(report, args.html)
        except NotImplementedError:
            print(f"\n(--html requested {args.html!r}, UNBUILT")
            return
        print(f"\nwrote HTML report to {args.html}")

def _cmd_synth(args: argparse.Namespace) -> None:
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
    parser = argparse.ArgumentParser(prog="navyra-profile")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyse = sub.add_parser("analyse")
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

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}")
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)
    
    except NotImplementedError as e:
        print(f"error: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
