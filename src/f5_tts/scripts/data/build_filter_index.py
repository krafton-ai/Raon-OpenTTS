#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build filter indices for TTS data filtering.

Reads pre-computed filter scores (DNSMOS, WER, VAD) from Arrow files,
identifies the worst-quality fraction, and saves keep-indices as JSON.

Usage:
    python -m f5_tts.scripts.data.build_filter_index \
        --score_dir data/filter_scores \
        --output_dir data/filter_indices \
        --method all \
        --ratio 0.15
"""

import argparse
import os
import json
import glob
import numpy as np
import pyarrow.ipc as ipc
from typing import List, Tuple, Optional, Dict


# Default dataset names (can be overridden via --datasets)
DATASETS = [
    "hifitts2",
    "libriheavy",
    "emilia-yodas2",
    "emilia",
    "voxpopuli",
    "gigaspeech",
    "youtube-chunks",
    "youtube",
    "peoples_speech_clean",
    "peoples_speech_dirty",
    "libritts-r",
    "spgispeech",
    "spgispeech2_cut",
]

# Column name candidates for each metric
CAND_DNSMOS = ["dnsmos_OVRL", "dnsmos"]
CAND_WER = ["wer"]
CAND_VAD = ["vad_ratio"]


def log(*a):
    print(*a, flush=True)


def read_arrow_col(path: str, cand_cols: List[str]) -> np.ndarray:
    """Read a single column from an Arrow dataset directory, streaming."""
    files = sorted(glob.glob(os.path.join(path, "data-*.arrow")))
    if not files:
        return np.array([], dtype=np.float32)

    # Handle multiple series (e.g. of-00005 and of-00007): use largest series
    series = {}
    for f in files:
        total = os.path.basename(f).split("-of-")[1].replace(".arrow", "")
        series.setdefault(total, []).append(f)
    if len(series) > 1:
        latest = max(series.keys(), key=lambda k: len(series[k]))
        files = sorted(series[latest])

    vals = []
    for f in files:
        with open(f, "rb") as fh:
            reader = ipc.RecordBatchStreamReader(fh)
            col = None
            for c in cand_cols:
                if c in reader.schema.names:
                    col = c
                    break
            if col is None:
                return np.array([], dtype=np.float32)
            for batch in reader:
                arr = batch.column(col)
                vals.append(arr.to_numpy(zero_copy_only=False).astype(np.float32))
    return np.concatenate(vals)


def select_worst_fraction_with_threshold(
    x: np.ndarray,
    frac: float,
    larger_is_worse: bool,
) -> Tuple[np.ndarray, Optional[float], int]:
    """Select the worst fraction of samples by score, handling boundary ties."""
    m = np.isfinite(x)
    finite_count = int(m.sum())
    if finite_count == 0:
        return np.zeros_like(x, dtype=bool), None, 0

    target_remove = int(finite_count * frac)
    finite_vals = x[m]

    if larger_is_worse:
        q = 1.0 - frac
        thresh = float(np.quantile(finite_vals, q))
        strictly_worse = m & (x > thresh)
        on_boundary = m & (x == thresh)
    else:
        q = frac
        thresh = float(np.quantile(finite_vals, q))
        strictly_worse = m & (x < thresh)
        on_boundary = m & (x == thresh)

    n_strict = int(strictly_worse.sum())
    n_boundary = int(on_boundary.sum())
    need_from_boundary = target_remove - n_strict

    mask = strictly_worse.copy()
    if need_from_boundary > 0 and n_boundary > 0:
        boundary_idx = np.where(on_boundary)[0]
        rng = np.random.RandomState(42)
        chosen = rng.choice(
            boundary_idx, size=min(need_from_boundary, n_boundary), replace=False
        )
        mask[chosen] = True

    return mask, thresh, finite_count


def argsort_rank(vals: np.ndarray) -> np.ndarray:
    """Compute ranks using argsort (int32 to save memory)."""
    n = len(vals)
    order = np.argsort(vals, kind="quicksort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(1, n + 1, dtype=np.float32)
    del order
    return ranks


def save_indices(out_dir: str, frac: float, name: str, mask: np.ndarray, total: int):
    """Write kept indices as a JSON array."""
    indices = np.where(mask)[0]
    n_kept = len(indices)
    out_path = os.path.join(
        out_dir, f"pool_indices_filter_remove_{int(frac * 100)}pct_{name}.json"
    )
    with open(out_path, "w") as f:
        f.write("[")
        for i, idx in enumerate(indices):
            if i > 0:
                f.write(",")
            f.write(str(int(idx)))
        f.write("]")
    log(
        f"[keep-{name}] kept={n_kept:,} ({n_kept / total * 100:.2f}% of pool) -> {out_path}"
    )


def load_metrics(score_dir: str, datasets: List[str]):
    """Load DNSMOS, WER, and VAD scores from Arrow files."""
    dnsmos_parts, wer_parts, vad_parts = [], [], []

    log("\n[metric] loading DNSMOS ...")
    for ds in datasets:
        p = os.path.join(score_dir, ds, "dnsmos")
        arr = read_arrow_col(p, CAND_DNSMOS)
        log(f"  {ds}: {len(arr):,}")
        dnsmos_parts.append(arr)
    dnsmos = np.concatenate(dnsmos_parts)
    del dnsmos_parts

    log("[metric] loading WER ...")
    for ds in datasets:
        p = os.path.join(score_dir, ds, "wer_result")
        arr = read_arrow_col(p, CAND_WER)
        log(f"  {ds}: {len(arr):,}")
        wer_parts.append(arr)
    wer = np.concatenate(wer_parts)
    del wer_parts

    log("[metric] loading VAD ...")
    for ds in datasets:
        p = os.path.join(score_dir, ds, "vad")
        arr = read_arrow_col(p, CAND_VAD)
        log(f"  {ds}: {len(arr):,}")
        vad_parts.append(arr)
    vad = np.concatenate(vad_parts)
    del vad_parts

    return dnsmos, wer, vad


def main():
    parser = argparse.ArgumentParser(
        description="Build filter indices for TTS data filtering."
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default="data/filter_scores",
        help="Directory containing per-dataset filter scores (default: data/filter_scores)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/filter_indices",
        help="Directory to save filter index files (default: data/filter_indices)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["dnsmos", "wer", "vad", "combined", "random", "all"],
        help="Filtering method to apply (default: all)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.15,
        help="Fraction of worst samples to remove (default: 0.15)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to process (default: built-in DATASETS list)",
    )
    args = parser.parse_args()

    frac = args.ratio
    datasets = args.datasets if args.datasets else DATASETS
    os.makedirs(args.output_dir, exist_ok=True)

    log(f"[config] ratio={frac}, method={args.method}")
    log(f"[config] score_dir={args.score_dir}")
    log(f"[config] output_dir={args.output_dir}")
    log(f"[config] datasets={datasets}")

    # Load all metrics
    dnsmos, wer, vad = load_metrics(args.score_dir, datasets)
    total = len(dnsmos)
    log(f"  total samples: {total:,}")

    assert len(dnsmos) == len(wer) == len(vad) == total, (
        f"Size mismatch: dnsmos={len(dnsmos)} wer={len(wer)} vad={len(vad)}"
    )

    log(
        f"\n[metric] non-NaN counts:",
        f"dnsmos={np.isfinite(dnsmos).sum():,}",
        f"wer={np.isfinite(wer).sum():,}",
        f"vad={np.isfinite(vad).sum():,}",
    )

    # Post-process WER: normalize to [0, 1] and fill NaN with 1.0
    high = np.isfinite(wer) & (wer > 1.0)
    wer[high] = wer[high] / 100.0
    wer[np.isfinite(wer) & (wer > 1.0)] = np.nan
    wer_nan_count = int((~np.isfinite(wer)).sum())
    wer[~np.isfinite(wer)] = 1.0
    log(f"[wer] filled {wer_nan_count:,} NaN values with 1.0")

    # Post-process VAD: normalize to [0, 1]
    big = np.isfinite(vad) & (vad > 1.0)
    vad[big] = vad[big] / 100.0

    methods = (
        ["dnsmos", "wer", "vad", "combined", "random"]
        if args.method == "all"
        else [args.method]
    )

    # Per-metric filtering
    if "dnsmos" in methods:
        worst_dnsmos, thr_dnsmos, n_dnsmos = select_worst_fraction_with_threshold(
            dnsmos, frac, larger_is_worse=False
        )
        good_dnsmos = np.isfinite(dnsmos) & (~worst_dnsmos)
        save_indices(args.output_dir, frac, "dnsmos", good_dnsmos, total)
        del good_dnsmos, worst_dnsmos

    if "wer" in methods:
        worst_wer, thr_wer, n_wer = select_worst_fraction_with_threshold(
            wer, frac, larger_is_worse=True
        )
        good_wer = np.isfinite(wer) & (~worst_wer)
        save_indices(args.output_dir, frac, "wer", good_wer, total)
        del good_wer, worst_wer

    if "vad" in methods:
        worst_vad, thr_vad, n_vad = select_worst_fraction_with_threshold(
            vad, frac, larger_is_worse=False
        )
        good_vad = np.isfinite(vad) & (~worst_vad)
        save_indices(args.output_dir, frac, "vad", good_vad, total)
        del good_vad, worst_vad

    if "combined" in methods:
        log("\n[combined] computing ranks ...")

        # dnsmos: higher is better, NaN -> -inf
        log("  ranking dnsmos...")
        d_vals = np.where(np.isfinite(dnsmos), dnsmos, np.float32(-np.inf))
        rank_d = argsort_rank(d_vals)
        del d_vals

        # wer: lower is better -> negate so higher rank = better
        log("  ranking wer...")
        w_vals = np.where(np.isfinite(wer), -wer, np.float32(-np.inf))
        rank_w = argsort_rank(w_vals)
        del w_vals

        # vad: higher is better, NaN -> -inf
        log("  ranking vad...")
        v_vals = np.where(np.isfinite(vad), vad, np.float32(-np.inf))
        rank_v = argsort_rank(v_vals)
        del v_vals

        log("  computing combined rank...")
        combined_rank = (rank_d + rank_w + rank_v) / 3.0
        del rank_d, rank_w, rank_v

        thresh_combined = float(np.quantile(combined_rank, frac))
        good_combined = combined_rank > thresh_combined
        del combined_rank
        save_indices(args.output_dir, frac, "combined", good_combined, total)
        del good_combined

    if "random" in methods:
        log(f"\n[random] removing {frac * 100:.0f}% randomly (seed=42) ...")
        rng = np.random.RandomState(42)
        n_keep = total - int(total * frac)
        random_mask = np.zeros(total, dtype=bool)
        random_mask[np.sort(rng.choice(total, size=n_keep, replace=False))] = True
        save_indices(args.output_dir, frac, "random", random_mask, total)

    # Save thresholds metadata
    thresholds_path = os.path.join(
        args.output_dir, f"thresholds_filter_remove_{int(frac * 100)}pct.json"
    )
    meta = {
        "frac_worst_removed": frac,
        "method": args.method,
        "datasets": datasets,
        "total_samples": total,
    }
    with open(thresholds_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"\n[meta] saved -> {thresholds_path}")

    log("\nDone.")


if __name__ == "__main__":
    main()
