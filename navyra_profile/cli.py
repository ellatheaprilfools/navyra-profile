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

def _cmd_analyse(args):
    prompts = []
    with open(args.file) as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj[args.field])

    report = analyse(prompts, radius=args.radius)
    print(report.summary())

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _cmd_analyse(args)

if __name__ == "__main__":
    main()
