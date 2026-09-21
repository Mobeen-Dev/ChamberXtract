#!/usr/bin/env python3
"""
vocab_cli.py - Extract word frequencies from a CSV column, edit them
interactively (delete / add), and save the curated keyword list.

Usage:
    # 1) Dump every word + its count to a CSV for review
    python vocab_cli.py extract data.csv --column company_name -o words.csv

    # 2) Interactively curate (type words to remove, then words to add)
    python vocab_cli.py edit words.csv -o keywords.json

    # One-shot, non-interactive editing (for scripting):
    python vocab_cli.py edit words.csv -o keywords.json \
        --remove "PVT,LTD,AND" --add "STEELWORKS,AGRO"
"""

import argparse
import sys

import pandas as pd

from vocab_core import apply_edits, save_curated, to_table, word_frequencies


def cmd_extract(args):
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    counts = word_frequencies(df, args.column)
    table = to_table(counts)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} distinct words -> {args.out}")
    print(table.head(20).to_string(index=False))


def cmd_edit(args):
    table = pd.read_csv(args.words_csv)
    counts = dict(zip(table["word"], table["count"]))

    if args.remove or args.add is not None:
        remove = {w.strip() for w in (args.remove or "").split(",") if w.strip()}
        add = {w.strip() for w in (args.add or "").split(",") if w.strip()}
    else:
        remove, add = _interactive(counts)

    from collections import Counter
    curated = apply_edits(Counter(counts), remove=remove, add=add)
    path = save_curated(curated.keys(), args.out, fmt=args.format)
    print(f"Removed {len(remove)}, added {len(add)}. "
          f"{len(curated)} words saved -> {path}")


def _interactive(counts):
    print(f"{len(counts)} words loaded. Sorted by frequency:\n")
    for w, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:>6}  {w}")
    print("\nType words to REMOVE, comma-separated (blank = none):")
    remove = {w.strip() for w in input("> ").split(",") if w.strip()}
    print("Type NEW words to ADD, comma-separated (blank = none):")
    add = {w.strip() for w in input("> ").split(",") if w.strip()}
    return remove, add


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pull word/count table from a CSV column")
    e.add_argument("csv")
    e.add_argument("--column", required=True)
    e.add_argument("-o", "--out", default="words.csv")
    e.set_defaults(func=cmd_extract)

    ed = sub.add_parser("edit", help="delete/add words, save curated list")
    ed.add_argument("words_csv", help="output of the extract step")
    ed.add_argument("-o", "--out", default="keywords.json")
    ed.add_argument("--format", choices=["json", "txt"], default="json")
    ed.add_argument("--remove", help="comma-separated words to drop (skips prompts)")
    ed.add_argument("--add", help="comma-separated words to add (skips prompts)")
    ed.set_defaults(func=cmd_edit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())