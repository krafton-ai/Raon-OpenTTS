#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate duration.json from actual audio file lengths.

Reads an Arrow dataset directory, computes actual durations from audio files,
and saves a duration.json mapping audio_path -> duration_in_seconds.

Usage:
    python -m f5_tts.scripts.data.prepare_duration \
        --dataset_dir data/metadata/gigaspeech \
        --audio_root data/raw/gigaspeech
"""

import argparse
import json
import os

import torchaudio
from datasets import load_from_disk
from tqdm import tqdm


def compute_duration(audio_path: str) -> float:
    """Compute audio duration in seconds using torchaudio."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate duration.json from actual audio file lengths."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to Arrow dataset directory",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default=None,
        help="Root directory for audio files. If not specified, audio_path values "
        "are treated as absolute paths.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio_path",
        help="Name of the column containing audio file paths (default: audio_path)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for duration.json (default: <dataset_dir>/duration.json)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.dataset_dir, "duration.json")

    print(f"Loading dataset from {args.dataset_dir} ...", flush=True)
    ds = load_from_disk(args.dataset_dir)
    print(f"  {len(ds)} samples loaded", flush=True)

    durations = {}
    errors = 0

    for i in tqdm(range(len(ds)), desc="Computing durations"):
        audio_rel = ds[i][args.audio_column]

        if args.audio_root:
            audio_abs = os.path.join(args.audio_root, audio_rel)
        else:
            audio_abs = audio_rel

        try:
            dur = compute_duration(audio_abs)
            durations[audio_rel] = round(dur, 4)
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  WARNING: Failed to read {audio_abs}: {e}", flush=True)
            if errors == 10:
                print("  (suppressing further warnings)", flush=True)

    print(f"\nComputed durations for {len(durations)} files ({errors} errors)", flush=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(durations, f)
    print(f"Saved -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
