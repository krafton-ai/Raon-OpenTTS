#!/usr/bin/env python3
"""
Re-compute WER on already-inferred wavs using the official seed-tts-eval method.
Matches BytedanceSpeech/seed-tts-eval exactly: transformers Whisper large-v3,
punctuation removal + lowercase (apostrophe kept). Uses Ray for multi-GPU parallelism.

Usage:
  python -m f5_tts.eval.run_wer_official \
      --wav_dir eval_outputs/my_model/wavs \
      --meta_lst /path/to/meta.lst \
      --num_gpus 8

  # Connect to existing Ray cluster
  python -m f5_tts.eval.run_wer_official ... --ray_address auto
"""

import argparse
import logging
import os
import string
from pathlib import Path

import numpy as np
import jiwer

logger = logging.getLogger(__name__)


PUNCTUATION_ALL = string.punctuation


def normalize_en(text, remove_apostrophe=False):
    for x in PUNCTUATION_ALL:
        if x == "'" and not remove_apostrophe:
            continue
        text = text.replace(x, "")
    text = text.replace("  ", " ")
    return text.lower()


def parse_meta_lst(meta_path):
    """Returns dict: gen_id -> gen_text"""
    samples = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 4:
                gen_id, _, _, gen_text = parts
            elif len(parts) == 5:
                gen_id, _, _, gen_text, _ = parts
            else:
                continue
            samples[gen_id] = gen_text
    return samples


def create_wer_actor_class():
    import ray

    @ray.remote(num_gpus=1)
    class WERActor:
        def __init__(self, remove_apostrophe: bool):
            import string
            import torch
            import soundfile as sf
            import scipy.signal
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            self.sf = sf
            self.scipy_signal = scipy.signal
            self.torch = torch
            self.remove_apostrophe = remove_apostrophe
            self.punctuation_all = string.punctuation

            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3", torch_dtype=torch.float32
            ).to("cuda")
            self.model.eval()
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            logger.info("[WERActor] transformers Whisper large-v3 loaded (float32)")

        def eval_batch(self, samples):
            """samples: list of (wav_path, gen_id, truth)"""
            results = []
            for wav_path, gen_id, truth in samples:
                try:
                    wav, sr = self.sf.read(wav_path)
                    if sr != 16000:
                        wav = self.scipy_signal.resample(wav, int(len(wav) * 16000 / sr))
                    input_features = self.processor(
                        wav, sampling_rate=16000, return_tensors="pt"
                    ).input_features.to("cuda")
                    with self.torch.no_grad():
                        predicted_ids = self.model.generate(
                            input_features, forced_decoder_ids=self.forced_decoder_ids
                        )
                    hypo_raw = self.processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]

                    truth_norm = normalize_en(truth, self.remove_apostrophe)
                    hypo_norm = normalize_en(hypo_raw, self.remove_apostrophe)

                    if not truth_norm.strip():
                        results.append({"gen_id": gen_id, "wer": None, "error": "empty_ref"})
                        continue

                    wer_val = float(jiwer.wer(truth_norm, hypo_norm))
                    results.append({
                        "gen_id": gen_id,
                        "truth": truth,
                        "hypo": hypo_raw,
                        "wer": wer_val,
                    })
                except Exception as e:
                    results.append({"gen_id": gen_id, "wer": None, "error": str(e)})
            return results

    return WERActor


def run_wer_ray(wav_dir, meta_path, num_gpus=8, remove_apostrophe=False):
    import ray

    label = "without apostrophe" if remove_apostrophe else "with apostrophe (official)"
    logger.info(f"[run_wer] {label}")
    logger.info(f"  wav_dir: {wav_dir}, num_gpus: {num_gpus}")

    id2text = parse_meta_lst(meta_path)
    wav_files = sorted(Path(wav_dir).glob("*.wav"))
    logger.info(f"  Found {len(wav_files)} wavs")

    samples = []
    for wav_path in wav_files:
        gen_id = wav_path.stem
        if gen_id in id2text:
            samples.append((str(wav_path), gen_id, id2text[gen_id]))
        else:
            logger.warning(f"  {gen_id} not in meta, skipping")

    WERActor = create_wer_actor_class()
    actors = [WERActor.remote(remove_apostrophe) for _ in range(num_gpus)]

    chunks = [[] for _ in range(num_gpus)]
    for i, s in enumerate(samples):
        chunks[i % num_gpus].append(s)

    futures = [a.eval_batch.remote(c) for a, c in zip(actors, chunks)]
    all_results = [r for batch in ray.get(futures) for r in batch]

    wer_values = [r["wer"] for r in all_results if r.get("wer") is not None]
    errors = [(r["gen_id"], r.get("error", "unknown")) for r in all_results if r.get("wer") is None]

    wer_mean = float(np.mean(wer_values)) if wer_values else None
    logger.info(f"\n{'=' * 60}")
    if wer_mean is not None:
        logger.info(f"WER ({label}): {wer_mean:.4f} ({wer_mean*100:.2f}%)  [{len(wer_values)} samples]")
    else:
        logger.info(f"WER ({label}): N/A (0 valid samples)")
    if errors:
        logger.info(f"Errors/skipped: {len(errors)}")
        for eid, emsg in errors[:5]:
            logger.info(f"  {eid}: {emsg}")
    logger.info(f"{'=' * 60}\n")

    return wer_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--meta_lst", required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--ray_address", type=str, default=None,
                        help="Ray cluster address. 'auto' to connect to existing cluster.")
    args = parser.parse_args()

    import ray
    ray_init_kwargs = {"ignore_reinit_error": True}
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    ray.init(**ray_init_kwargs)

    wer_official = run_wer_ray(args.wav_dir, args.meta_lst, args.num_gpus, remove_apostrophe=False)
    wer_no_apos = run_wer_ray(args.wav_dir, args.meta_lst, args.num_gpus, remove_apostrophe=True)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if wer_official is not None and wer_no_apos is not None:
        logger.info(f"  Official (apostrophe kept):   {wer_official*100:.4f}%")
        logger.info(f"  Apostrophe removed:           {wer_no_apos*100:.4f}%")
        logger.info(f"  Difference:                   {(wer_no_apos - wer_official)*100:+.4f}%")
    else:
        logger.info(f"  Official: {wer_official}")
        logger.info(f"  No apostrophe: {wer_no_apos}")
    logger.info("=" * 60)

    ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
