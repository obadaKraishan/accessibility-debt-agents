"""
Shared argument parsing, so every script accepts the same core flags.

    --limit N      process only the first N units (fast pilot runs)
    --restart      ignore any checkpoint and start clean
    --quiet        suppress console output (the log file still gets everything)
    --seed N       override the random seed
    --out PATH     override the output directory
"""

from __future__ import annotations

import argparse
from typing import Sequence


def base_parser(description: str, epilog: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N units (pilot run)")
    parser.add_argument("--restart", action="store_true",
                        help="ignore any saved checkpoint and start from scratch")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress console output; the log file still records everything")
    parser.add_argument("--seed", type=int, default=20260908,
                        help="random seed (default: 20260908)")
    parser.add_argument("--out", type=str, default=None,
                        help="override the output directory for this script")
    return parser


def parse(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None):
    return parser.parse_args(argv)
