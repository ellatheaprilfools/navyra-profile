"""
navyra_profile.cli — command-line interface.  [TO BUILD]

PROJECT TASK 2 (part of packaging). Target UX:

  navyra-profile analyse traffic.jsonl --field prompt --radius 6
  navyra-profile analyse traffic.jsonl --html report.html
  navyra-profile synth --n 10000 --template-share 0.4 -o synth.jsonl

Use argparse or typer. JSONL in, report out. Errors should be friendly.
"""

import argparse
from .analyser import analyse
from .fingerprint import Fingerprinter
import json
import sys

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog= "navyra-profile")
    sub = parser.add_subparsers(dest="command", required=True) #has to add analyse or synth
    
    p_analyse = sub.add_parser("analyse")
    p_analyse.add_argument("file", help="path to a JSONL file of prompts")

    p_analyse.add_argument( #optional
        "--field",
        default="prompt",
        help="JSON field containing the prompt text",
    )

    p_analyse.add_argument(
        "--radius", 
        type=int,
        default=6,#defaulted to 6
        help="Hamming-distance threshold for a semantic hit",
    )

    return parser

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _cmd_analyse(args)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}")
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
