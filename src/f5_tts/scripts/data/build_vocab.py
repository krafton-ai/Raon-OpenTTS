#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a merged vocabulary file from Arrow datasets.

Reads the text column from each dataset, extracts unique characters,
sorts them, and saves to vocab.txt.

Usage:
    python -m f5_tts.scripts.data.build_vocab \
        --dataset_dir data/metadata \
        --output data/metadata/vocab.txt
"""

import argparse
import os

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(
        description="Build a merged vocab.txt from Arrow datasets."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory containing Arrow dataset subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for vocab.txt (default: <dataset_dir>/vocab.txt)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column (default: text)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset names to include (default: all subdirectories)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.dataset_dir, "vocab.txt")

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(
            d
            for d in os.listdir(args.dataset_dir)
            if os.path.isdir(os.path.join(args.dataset_dir, d))
        )

    all_chars = set()

    for name in dataset_names:
        ds_path = os.path.join(args.dataset_dir, name)
        print(f"[{name}] Loading ...", flush=True)
        try:
            ds = load_from_disk(ds_path)
        except Exception as e:
            print(f"  WARNING: Failed to load {ds_path}: {e}", flush=True)
            continue

        if args.text_column not in ds.column_names:
            print(
                f"  WARNING: Column '{args.text_column}' not found in {name}, skipping",
                flush=True,
            )
            continue

        texts = ds[args.text_column]
        chars_before = len(all_chars)
        for text in texts:
            if text:
                all_chars.update(text)
        print(
            f"  {len(texts)} texts, {len(all_chars) - chars_before} new chars",
            flush=True,
        )

    vocab = sorted(all_chars)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for char in vocab:
            f.write(char + "\n")

    print(f"\nTotal unique characters: {len(vocab)}", flush=True)
    print(f"Saved -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
