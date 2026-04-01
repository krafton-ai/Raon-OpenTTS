# eval_all_unified.py
# Unified WER / SIM evaluation script
# - LibriSpeech (test-clean, test-other)
# - STspeech
# - Expresso
# - HF eval sets (CMU_Arctic, L2Arctic, ... EmoV_DB, etc.)

import argparse
import json
import logging
import os
import sys
import tempfile
import multiprocessing as mp
from collections import defaultdict

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torchaudio
import soundfile as sf
from datasets import load_from_disk, concatenate_datasets

from .utils_eval import (
    run_asr_wer,
    run_sim,
    get_librispeech_test,  # used in existing Libri eval
)
MAX_SEC = 30.0

def wav_duration_sec(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)

def is_over_max_sec(path: str, max_sec: float = MAX_SEC) -> bool:
    try:
        return wav_duration_sec(path) > max_sec
    except Exception:
        return True
# ----------------------------------
# Dataset / domain / type definitions
# ----------------------------------

DATASETS_INFO = {
    # LibriSpeech family (local)
    "test-clean": {
        "domain": "CLEAN",
        "type": "librispeech",
        "metalst": "data/librispeech_pc_test_clean_cross_sentence.lst",
        "audio_root": "data/librispeech/test-clean",
    },
    "test-other": {
        "domain": "NOISY",
        "type": "librispeech",
        "metalst": "data/librispeech_pc_test_other_cross_sentence.lst",
        "audio_root": "data/librispeech/test-other",
    },

    # STspeech (local, any format)
    "STspeech": {
        "domain": "CLEAN",
        "type": "stspeech",
        "metalst": "data/stspeech_500_balanced.lst",
        "audio_root": "data/stspeech",
    },

    # ----- HF Eval sets -----
    "CMU_Arctic": {"domain": "CLEAN", "type": "hf"},
    "L2Arctic": {"domain": "CLEAN", "type": "hf"},
    "ami-ihm": {"domain": "WILD", "type": "hf"},
    "switchboard": {"domain": "WILD", "type": "hf"},
    "tedlium3_test": {"domain": "NOISY", "type": "hf"},
    "crema-d": {"domain": "EMOTIONAL", "type": "hf"},
    "EmoV_DB": {"domain": "EMOTIONAL", "type": "hf"},
    "expresso": {"domain": "EMOTIONAL", "type": "hf"},
    "vctk": {"domain": "CLEAN", "type": "hf"},
}

HF_DATASETS = [k for k, v in DATASETS_INFO.items() if v["type"] == "hf"]

# HF audio column mapping (for HF datasets)
DATASET_AUDIO_COL = {
    "CMU_Arctic": "audio",
    "L2Arctic": "audio",
    "ami-ihm": "audio",
    "crema-d": "audio",
    "EmoV_DB": "audio",
    "switchboard": "audio",
    "expresso": "audio",
    "vctk": "audio",
    "tedlium3_test": "context",  # audio is stored in this column
}


# ----------------------------------
# Arguments
# ----------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag for generated wav directory (e.g., 40k-wer90_1.2M)",
    )

    parser.add_argument(
        "--gen_root",
        type=str,
        default="output/evaluation",
        help="Root directory for generated wavs (output/evaluation/{DOMAIN}/{dataset}/{tag})",
    )

    # HF LST / dataset root
    parser.add_argument(
        "--lst_root",
        type=str,
        required=True,
        help="Root directory containing HF eval LST files",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing HF Arrow datasets",
    )

    parser.add_argument(
        "--gpu_nums",
        type=int,
        default=1,
        help="Number of GPUs to use (currently only 1 is used)",
    )

    parser.add_argument(
        "--asr_ckpt_dir",
        type=str,
        default="",
        help="ASR checkpoint dir for WER (empty string uses HF model from utils_eval)",
    )
    parser.add_argument(
        "--wavlm_ckpt_dir",
        type=str,
        default="wavlm_large_finetune.pth",
    )

    # Whether to skip datasets that already have results
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip datasets that already have _sim_results.jsonl / _wer_results.jsonl",
    )

    return parser.parse_args()


# ----------------------------------
# HF dataset helper (based on eval_all_evals.py)
# ----------------------------------

_dataset_cache = {}
REF_CACHE_DIR = os.path.join(tempfile.gettempdir(), "hf_eval_refs")
os.makedirs(REF_CACHE_DIR, exist_ok=True)


def get_hf_dataset(ds_name: str, dataset_root: str):
    if ds_name in _dataset_cache:
        return _dataset_cache[ds_name]

    ds_path = os.path.join(dataset_root, ds_name)
    logger.info(f"> [HF] Loading dataset from: {ds_path}")
    ds = load_from_disk(ds_path)
    if hasattr(ds, "keys"):
        ds = concatenate_datasets([ds[s] for s in ds.keys()])

    _dataset_cache[ds_name] = ds
    return ds


def ref_id_to_wav_path(ref_id: str, dataset_root: str, target_sr: int = 16000) -> str:
    """
    ref_id: format 'CMU_Arctic:839'.
    Extracts audio at the given index from HF dataset, saves to REF_CACHE_DIR/{ds_name}_{idx}.wav, and returns the path.
    """
    ds_name, idx_str = ref_id.split(":")
    idx = int(idx_str)

    if ds_name not in DATASET_AUDIO_COL:
        raise ValueError(f"Unknown HF dataset in ref_id: {ref_id}")

    audio_col = DATASET_AUDIO_COL[ds_name]
    ds = get_hf_dataset(ds_name, dataset_root)

    if idx < 0 or idx >= len(ds):
        raise IndexError(f"Index out of range for {ds_name}: {idx}")

    out_path = os.path.join(REF_CACHE_DIR, f"{ds_name}_{idx}.wav")
    if os.path.exists(out_path):
        return out_path

    ex = ds[idx]
    audio = ex[audio_col]
    arr = audio["array"]
    sr = audio["sampling_rate"]
    if isinstance(sr, (list, tuple)):
        sr = sr[0]

    wav = torch.from_numpy(arr).float().unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav_np = wav.squeeze(0).cpu().numpy().astype("float32")
    sf.write(out_path, wav_np, target_sr)
    return out_path


def build_testset_hf(metalst_path, gen_wav_dir, gpus, dataset_name, dataset_root):
    """
    For HF datasets: ref_id -> HF audio, gen_id -> gen wav, uses gen_text only.
    Only includes samples where gen wav actually exists.
    Returns: [(gpu_id, [(gen_wav, ref_wav, text), ...])]
    """
    samples = []

    if not os.path.exists(metalst_path):
        logger.info(f"[HF] metalst not found: {metalst_path}")
        return []

    with open(metalst_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        ref_id, _, ref_text, gen_id, _, gen_text = parts

        gen_fname = gen_id.replace(":", "_") + ".wav"
        gen_wav = os.path.join(gen_wav_dir, gen_fname)
        if not os.path.exists(gen_wav):
            continue

        ref_wav = ref_id_to_wav_path(ref_id, dataset_root, target_sr=16000)
        text = gen_text
        samples.append((gen_wav, ref_wav, text))

    if not samples:
        return []

    # Assuming single GPU usage
    return [(gpus[0], samples)]


# ----------------------------------
# STspeech / Expresso helper
# ----------------------------------

def build_testset_stspeech(metalst_path, gen_wav_dir, gpus, audio_root):
    """
    STspeech LST: ref_id, ref_dur, ref_text, gen_id, gen_dur, gen_text
    - ref_id: assumed to be audio file basename (without extension) -> audio_root/{ref_id}.wav
    - Only uses samples where gen wav exists
    """
    samples = []
    if not os.path.exists(metalst_path):
        logger.info(f"[STspeech] metalst not found: {metalst_path}")
        return []

    with open(metalst_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        ref_id, _, ref_text, gen_id, _, gen_text = parts[:6]

        ref_path = os.path.join(audio_root, ref_id + ".wav")
        gen_path = os.path.join(gen_wav_dir, gen_id + ".wav")
        if not (os.path.exists(ref_path) and os.path.exists(gen_path)):
            continue

        truth = gen_text
        samples.append((gen_path, ref_path, truth))

    if not samples:
        return []

    # Single GPU
    return [(gpus[0], samples)]


# ----------------------------------
# Load metrics from existing result jsonl
# ----------------------------------

def load_existing_metrics_from_jsonl(result_path, metric):
    """
    Reads metric values from _sim_results.jsonl or _wer_results.jsonl.
    Ignores summary lines like 'SIM: 0.12345' at the end.
    """
    metrics = []
    if not os.path.exists(result_path):
        return metrics

    with open(result_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # Skip non-JSON summary lines at the end
            if not ln.startswith("{"):
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if metric in obj:
                metrics.append(obj[metric])
    return metrics


# ----------------------------------
# Evaluate a single metric (SIM or WER)
# ----------------------------------

def run_one_metric(metric, args, per_dataset, per_domain, overall):
    assert metric in ["sim", "wer"]
    logger.info(f"\n================ {metric.upper()} EVAL START ================\n")

    lang = "en"
    gpus = [0]  # assuming gpu_nums=1
    asr_ckpt_dir = args.asr_ckpt_dir
    wavlm_ckpt_dir = args.wavlm_ckpt_dir

    for ds_name, info in DATASETS_INFO.items():
        domain = info["domain"]
        ds_type = info["type"]

        # gen dir: /gen_root/{DOMAIN}/{dataset}/{tag}
        gen_dir = os.path.join(args.gen_root, domain, ds_name, args.tag)
        if not os.path.isdir(gen_dir):
            logger.info(f"[WARN] Gen dir not found for {ds_name}: {gen_dir}")
            continue

        # Result file path
        result_path = os.path.join(gen_dir, f"_{metric}_results.jsonl")

        expected_n = expected_sample_count_for_dataset(ds_name, info, args, gpus, metric)

        # Result file path
        result_path = os.path.join(gen_dir, f"_{metric}_results.jsonl")

        # Skip if results already exist and recomputation is not needed:
        if args.skip_existing and os.path.exists(result_path):
            existing_n = count_existing_metric_lines(result_path, metric)

            if expected_n > 0 and existing_n == expected_n:
                logger.info(f"[SKIP] {ds_name} ({metric}): existing_n({existing_n}) == expected_n({expected_n}) -> {result_path}")

                # Load metric values from existing results for statistics
                metrics = load_existing_metrics_from_jsonl(result_path, metric)
                if not metrics:
                    logger.info(f"[WARN] Could not find {metric} metric in existing result file: {result_path}")
                    # File may be corrupted or malformed; fall through to re-evaluation
                else:
                    value = float(np.mean(metrics))
                    value_round = round(value, 5)
                    n = len(metrics)

                    # Record per-dataset results
                    if ds_name not in per_dataset:
                        per_dataset[ds_name] = {
                            "domain": domain,
                            "sim": None,
                            "wer": None,
                            "n_sim": 0,
                            "n_wer": 0,
                        }
                    per_dataset[ds_name][metric] = value_round
                    per_dataset[ds_name][f"n_{metric}"] = n

                    # Record per-domain results (sample-level)
                    per_domain[domain]["metrics"][metric].extend(metrics)

                    # Record overall results
                    overall[metric].extend(metrics)

                    logger.info(f"Total {n} samples (loaded from existing results)")
                    logger.info(f"{metric.upper()} ({ds_name}): {value_round}  [from existing jsonl]\n")
                    continue

            # Re-evaluate if count is insufficient or mismatched
            logger.info(
                f"[RE-RUN] {ds_name} ({metric}): existing_n({existing_n}) != expected_n({expected_n}) "
                f"-> recompute and overwrite {result_path}"
            )

        logger.info(f"===== DATASET: {ds_name} | DOMAIN: {domain} | TYPE: {ds_type} =====")
        logger.info(f"GEN : {gen_dir}")

        # build test_set depending on type
        if ds_type == "hf":
            metalst = os.path.join(args.lst_root, f"{ds_name}_zero_shot_3to30_unique500.lst")
            logger.info(f"LST : {metalst}")
            test_set = build_testset_hf(
                metalst_path=metalst,
                gen_wav_dir=gen_dir,
                gpus=gpus,
                dataset_name=ds_name,
                dataset_root=args.dataset_root,
            )

        elif ds_type == "librispeech":
            metalst = info["metalst"]
            audio_root = info["audio_root"]
            logger.info(f"LST (Libri): {metalst}")
            logger.info(f"Ref root   : {audio_root}")
            # Use existing get_librispeech_test (gen_wav_dir + metalst + librispeech root)
            test_set = get_librispeech_test(metalst, gen_dir, gpus, audio_root)

        elif ds_type == "stspeech":
            metalst = info["metalst"]
            audio_root = info["audio_root"]
            logger.info(f"LST (STspeech): {metalst}")
            logger.info(f"Ref root      : {audio_root}")
            test_set = build_testset_stspeech(metalst, gen_dir, gpus, audio_root)

        else:
            logger.info(f"[WARN] Unknown dataset type for {ds_name}: {ds_type}")
            continue

        if not test_set:
            logger.info(f"[WARN] No samples for {ds_name} ({metric})")
            continue

        full_results = []
        metrics = []

        if metric == "wer":
            with mp.Pool(processes=len(gpus)) as pool:
                mp_args = [
                    (rank, lang, sub_test_set, asr_ckpt_dir)
                    for (rank, sub_test_set) in test_set
                ]
                results = pool.map(run_asr_wer, mp_args)
                for r in results:
                    full_results.extend(r)
        else:  # sim
            with mp.Pool(processes=len(gpus)) as pool:
                mp_args = [
                    (rank, sub_test_set, wavlm_ckpt_dir)
                    for (rank, sub_test_set) in test_set
                ]
                results = pool.map(run_sim, mp_args)
                for r in results:
                    full_results.extend(r)

        for line in full_results:
            if metric not in line:
                continue
            metrics.append(line[metric])

        if not metrics:
            logger.info(f"[WARN] No {metric} metrics for {ds_name}")
            continue

        value = float(np.mean(metrics))
        value_round = round(value, 5)
        n = len(metrics)

        # Record per-dataset results
        if ds_name not in per_dataset:
            per_dataset[ds_name] = {
                "domain": domain,
                "sim": None,
                "wer": None,
                "n_sim": 0,
                "n_wer": 0,
            }
        per_dataset[ds_name][metric] = value_round
        per_dataset[ds_name][f"n_{metric}"] = n

        # Record per-domain results (sample-level)
        per_domain[domain]["metrics"][metric].extend(metrics)

        # Record overall results
        overall[metric].extend(metrics)

        # Save per-dataset jsonl
        with open(result_path, "w", encoding="utf-8") as f:
            for line in full_results:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.write(f"\n{metric.upper()}: {value_round}\n")

        logger.info(f"Total {n} samples")
        logger.info(f"{metric.upper()} ({ds_name}): {value_round}")
        logger.info(f"{metric.upper()} results saved to {result_path}\n")
def expected_sample_count_for_dataset(ds_name: str, info: dict, args, gpus, metric: str) -> int:
    """
    Computes the expected number of samples to evaluate based on gen_dir + metalst + (audio_root if needed).
    test_set format: [(gpu_id, [(gen, ref, text), ...])]
    """
    domain = info["domain"]
    ds_type = info["type"]
    gen_dir = os.path.join(args.gen_root, domain, ds_name, args.tag)

    if not os.path.isdir(gen_dir):
        return 0

    if ds_type == "hf":
        metalst = os.path.join(args.lst_root, f"{ds_name}_zero_shot_3to30_unique500.lst")
        # HF ref extraction can be expensive, so estimate count based on "gen wav exists + length limit" only
        # (assumes ref can always be extracted; idx out-of-range will fail at eval time)
        if not os.path.exists(metalst):
            return 0
        cnt = 0
        with open(metalst, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 6:
                    continue
                ref_id, _, ref_text, gen_id, _, gen_text = parts
                gen_fname = gen_id.replace(":", "_") + ".wav"
                gen_wav = os.path.join(gen_dir, gen_fname)
                if not os.path.exists(gen_wav):
                    continue
                cnt += 1
        return cnt

    elif ds_type == "librispeech":
        metalst = info["metalst"]
        audio_root = info["audio_root"]
        test_set = get_librispeech_test(metalst, gen_dir, gpus, audio_root)
        return sum(len(sub) for _, sub in test_set) if test_set else 0

    elif ds_type == "stspeech":
        metalst = info["metalst"]
        audio_root = info["audio_root"]
        test_set = build_testset_stspeech(metalst, gen_dir, gpus, audio_root)
        return sum(len(sub) for _, sub in test_set) if test_set else 0

    else:
        return 0
def count_existing_metric_lines(result_path: str, metric: str) -> int:
    """
    Counts the number of JSON lines containing the metric key in _sim_results.jsonl / _wer_results.jsonl.
    (Excludes the summary line 'SIM: ...' at the end)
    """
    if not os.path.exists(result_path):
        return 0

    n = 0
    with open(result_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or not ln.startswith("{"):
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if metric in obj:
                n += 1
    return n
# ----------------------------------
# Main
# ----------------------------------

def main():
    args = get_args()

    # Result storage structures
    per_dataset = {}  # {ds_name: {"domain":..., "sim":..., "wer":..., "n_sim":..., "n_wer":...}}
    per_domain = defaultdict(lambda: {"metrics": {"sim": [], "wer": []}})
    overall = {"sim": [], "wer": []}

    # SIM
    run_one_metric("sim", args, per_dataset, per_domain, overall)
    # WER
    run_one_metric("wer", args, per_dataset, per_domain, overall)

    # Summary output directory
    summary_dir = os.path.join(args.gen_root, f"eval_summary_{args.tag}")
    os.makedirs(summary_dir, exist_ok=True)

    # 1) Per-dataset results
    dataset_json = os.path.join(summary_dir, "per_dataset.json")
    with open(dataset_json, "w", encoding="utf-8") as f:
        json.dump(per_dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"\n[Saved] per-dataset results: {dataset_json}")

    # 2) Per-domain averages (sample-level mean)
    domain_summary = {}
    for domain, d in per_domain.items():
        sims = d["metrics"]["sim"]
        wers = d["metrics"]["wer"]
        sim_mean = round(float(np.mean(sims)), 5) if sims else None
        wer_mean = round(float(np.mean(wers)), 5) if wers else None
        domain_summary[domain] = {
            "sim_mean": sim_mean,
            "wer_mean": wer_mean,
            "sim_n": len(sims),
            "wer_n": len(wers),
        }

    domain_json = os.path.join(summary_dir, "per_domain.json")
    with open(domain_json, "w", encoding="utf-8") as f:
        json.dump(domain_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"[Saved] per-domain summary: {domain_json}")

    # 3) Overall averages (across all samples)
    overall_summary = {}
    for metric in ["sim", "wer"]:
        vals = overall[metric]
        overall_summary[metric] = {
            "mean": round(float(np.mean(vals)), 5) if vals else None,
            "n": len(vals),
        }

    overall_json = os.path.join(summary_dir, "overall.json")
    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"[Saved] overall summary: {overall_json}")

    # Print console summary
    logger.info("\n========== DOMAIN SUMMARY ==========")
    for domain, vals in domain_summary.items():
        logger.info(
            f"{domain:10s} | SIM: {vals['sim_mean']} (n={vals['sim_n']})"
            f" | WER: {vals['wer_mean']} (n={vals['wer_n']})"
        )
    logger.info("====================================")

    logger.info("\n========== OVERALL SUMMARY =========")
    for metric, vals in overall_summary.items():
        logger.info(f"{metric.upper():4s} | mean: {vals['mean']} (n={vals['n']})")
    logger.info("====================================")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mp.set_start_method("spawn")
    main()
