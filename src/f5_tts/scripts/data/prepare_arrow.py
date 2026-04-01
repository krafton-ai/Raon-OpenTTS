#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert raw audio + text data into HuggingFace Arrow dataset format.

Expects each dataset subdirectory under --input_dir to contain:
  - Audio files (.wav, .flac, .mp3, .ogg)
  - A metadata file (metadata.csv or metadata.tsv) with columns: audio_path, text

Usage:
    python -m f5_tts.scripts.data.prepare_arrow \
        --input_dir data/raw \
        --output_dir data/metadata
"""

import argparse
import csv
import os
from pathlib import Path

import torchaudio
from datasets import Dataset


def get_audio_duration(audio_path: str) -> float:
    """Compute audio duration in seconds using torchaudio."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def find_metadata_file(dataset_dir: str) -> str:
    """Find the metadata file in a dataset directory."""
    for name in ["metadata.csv", "metadata.tsv", "metadata.txt"]:
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No metadata file found in {dataset_dir}. "
        "Expected metadata.csv, metadata.tsv, or metadata.txt"
    )


def load_metadata(metadata_path: str):
    """Load metadata from CSV/TSV file. Returns list of (audio_path, text) tuples."""
    delimiter = "\t" if metadata_path.endswith(".tsv") else ","
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            audio_col = None
            text_col = None
            for key in row:
                kl = key.lower().strip()
                if kl in ("audio_path", "audio", "file", "filename", "path", "wav"):
                    audio_col = key
                if kl in ("text", "transcript", "transcription", "sentence"):
                    text_col = key
            if audio_col is None or text_col is None:
                raise ValueError(
                    f"Could not find audio/text columns in {metadata_path}. "
                    f"Available columns: {list(row.keys())}"
                )
            entries.append((row[audio_col], row[text_col]))
    return entries


def process_dataset(input_dir: str, output_dir: str, dataset_name: str):
    """Process a single dataset directory into Arrow format."""
    dataset_dir = os.path.join(input_dir, dataset_name)
    out_dir = os.path.join(output_dir, dataset_name)

    print(f"[{dataset_name}] Processing ...", flush=True)

    metadata_path = find_metadata_file(dataset_dir)
    entries = load_metadata(metadata_path)
    print(f"  Found {len(entries)} entries in metadata", flush=True)

    audio_paths = []
    texts = []
    durations = []
    skipped = 0

    for audio_rel, text in entries:
        audio_abs = os.path.join(dataset_dir, audio_rel)
        if not os.path.exists(audio_abs):
            skipped += 1
            continue
        try:
            duration = get_audio_duration(audio_abs)
        except Exception as e:
            print(f"  WARNING: Failed to read {audio_abs}: {e}", flush=True)
            skipped += 1
            continue

        audio_paths.append(audio_rel)
        texts.append(text)
        durations.append(duration)

    if skipped > 0:
        print(f"  Skipped {skipped} entries (missing or unreadable audio)", flush=True)

    ds = Dataset.from_dict(
        {
            "audio_path": audio_paths,
            "text": texts,
            "duration": durations,
        }
    )

    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    print(
        f"  Saved {len(ds)} samples to {out_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw audio+text data into HuggingFace Arrow format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing raw dataset subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Arrow datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset names to process (default: all subdirectories)",
    )
    args = parser.parse_args()

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(
            d
            for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d))
        )

    print(f"Processing {len(dataset_names)} datasets ...", flush=True)
    for name in dataset_names:
        try:
            process_dataset(args.input_dir, args.output_dir, name)
        except Exception as e:
            print(f"[{name}] ERROR: {e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
