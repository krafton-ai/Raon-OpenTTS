#!/usr/bin/env python3
"""
Unified TTS inference + evaluation (WER + Speaker Similarity) using Ray.
Fixed to seed-tts-eval English test set.

Usage:
  python scripts/run_infer_eval.py --ckpt checkpoints/640k-6node-norm-noseed/model_950000.pt
  python scripts/run_infer_eval.py --ckpt /path/to/model.pt --config src/configs/03b.yaml --num_gpus 4
"""

import argparse
import json
import logging
import os
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─── Path setup ──────────────────────────────────────────────
PROJ_ROOT = Path(str(files("f5_tts").joinpath("../.."))).resolve()

# Monkeypatch torchaudio for compat (2.10+ removed old backends, torchcodec needs FFmpeg)
import torch
import torchaudio
import soundfile as sf_mod

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

_original_ta_load = torchaudio.load
def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Fallback to soundfile if torchcodec fails."""
    try:
        return _original_ta_load(filepath, *args, **kwargs)
    except (ImportError, RuntimeError, OSError):
        data, sr = sf_mod.read(str(filepath), dtype="float32")
        wav = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
        return wav, sr

torchaudio.load = _patched_torchaudio_load

META_LST = os.environ.get("EVAL_META_LST", str(PROJ_ROOT / "eval" / "meta.lst"))
WAVLM_CKPT = os.environ.get("EVAL_WAVLM_CKPT", str(PROJ_ROOT / "checkpoints" / "wavlm_large_finetune.pth"))
DEFAULT_CONFIG = str(files("f5_tts").joinpath("configs/03b.yaml"))


# ─── Parse meta.lst ─────────────────────────────────────────
def parse_meta_lst(meta_path: str):
    """Returns list of (gen_id, ref_text, ref_audio_abs, gen_text)"""
    base_dir = os.path.dirname(os.path.abspath(meta_path))
    samples = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 4:
                gen_id, ref_text, ref_audio, gen_text = parts
            elif len(parts) == 5:
                gen_id, ref_text, ref_audio, gen_text, _ = parts
            else:
                continue
            if not os.path.isabs(ref_audio):
                ref_audio = os.path.join(base_dir, ref_audio)
            samples.append((gen_id, ref_text, ref_audio, gen_text))
    return samples


# ─── Ray Actor: Inference ────────────────────────────────────
def create_infer_actor_class():
    import ray

    @ray.remote(num_gpus=1)
    class InferActor:
        def __init__(self, config_path: str, ckpt_path: str):
            import torch
            from omegaconf import OmegaConf
            import hydra
            import soundfile as sf
            from ema_pytorch import EMA
            from f5_tts.model import CFM
            from f5_tts.infer.utils_infer import (
                load_vocoder, load_model, infer_process,
                target_rms, cross_fade_duration, nfe_step,
                cfg_strength, sway_sampling_coef, speed, fix_duration,
            )
            from f5_tts.model.utils import get_tokenizer

            self.sf = sf
            self.torch = torch
            self.infer_process = infer_process
            self.target_rms = target_rms
            self.cross_fade_duration = cross_fade_duration
            self.nfe_step = nfe_step
            self.cfg_strength = cfg_strength
            self.sway_sampling_coef = sway_sampling_coef
            self.speed = speed
            self.fix_duration = fix_duration

            self.device = torch.device("cuda:0")

            # Config
            model_cfg = OmegaConf.load(config_path)
            # Override flash_attn to torch for inference (flash_attn not always installed)
            if OmegaConf.select(model_cfg, "model.arch.attn_backend") == "flash_attn":
                OmegaConf.update(model_cfg, "model.arch.attn_backend", "torch")
            model_cls = hydra.utils.get_class(
                f"f5_tts.model.{model_cfg.model.backbone}"
            )
            model_arc = model_cfg.model.arch
            mel_spec_cfg = model_cfg.model.mel_spec
            self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
            self.model_cfg = model_cfg

            # Tokenizer
            tokenizer_name = model_cfg.model.tokenizer
            tokenizer_path = model_cfg.model.tokenizer_path

            paths = tokenizer_path.split("|")
            char_maps = [get_tokenizer(p, tokenizer_name)[0] for p in paths]
            all_tokens = {tok for cmap in char_maps for tok in cmap.keys()}
            vocab_char_map = {tok: idx for idx, tok in enumerate(sorted(all_tokens))}
            vocab_size = len(vocab_char_map)

            # Model
            if ckpt_path.endswith(".safetensors"):
                model = load_model(
                    model_cls=model_cls,
                    model_cfg=model_arc,
                    ckpt_path=ckpt_path,
                    vocab_char_map=vocab_char_map,
                    vocab_size=vocab_size,
                    mel_spec_type=self.mel_spec_type,
                    use_ema=True,
                    device=str(self.device),
                )
            else:
                model = CFM(
                    transformer=model_cls(
                        **model_arc,
                        text_num_embeds=vocab_size,
                        mel_dim=mel_spec_cfg.n_mel_channels,
                    ),
                    mel_spec_kwargs=mel_spec_cfg,
                    vocab_char_map=vocab_char_map,
                ).to(self.device)

                ema = EMA(model, include_online_model=False).to(self.device)
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "ema_model_state_dict" not in ckpt:
                    raise RuntimeError("Checkpoint does not contain 'ema_model_state_dict'.")
                ema.load_state_dict(ckpt["ema_model_state_dict"])
                for key, param in ema.ema_model.state_dict().items():
                    model.state_dict()[key].copy_(param)

            model.eval()
            self.model = model

            # Vocoder
            self.vocoder = load_vocoder(
                vocoder_name=self.mel_spec_type,
                is_local=model_cfg.model.vocoder.is_local,
                local_path=model_cfg.model.vocoder.local_path,
            )
            try:
                self.target_sample_rate = model.mel_spec.target_sample_rate
            except AttributeError:
                self.target_sample_rate = 22050

            logger.info(f"[InferActor] Ready on {self.device}, vocoder={self.mel_spec_type}, sr={self.target_sample_rate}")

        def infer_batch(self, samples, output_dir):
            """samples: list of (gen_id, ref_text, ref_audio, gen_text)"""
            results = []
            for i, (gen_id, ref_text, ref_audio, gen_text) in enumerate(samples):
                out_path = os.path.join(output_dir, f"{gen_id}.wav")
                if os.path.exists(out_path):
                    results.append({"gen_id": gen_id, "status": "skipped"})
                    continue
                try:
                    audio_segment, final_sr, _ = self.infer_process(
                        ref_audio,
                        ref_text.lower(),
                        gen_text.lower(),
                        self.model,
                        self.vocoder,
                        mel_spec_type=self.mel_spec_type,
                        target_rms=self.target_rms,
                        cross_fade_duration=self.cross_fade_duration,
                        nfe_step=self.nfe_step,
                        cfg_strength=self.cfg_strength,
                        sway_sampling_coef=self.sway_sampling_coef,
                        speed=self.speed,
                        fix_duration=self.fix_duration,
                        device=self.device,
                    )
                    self.sf.write(out_path, audio_segment, final_sr)
                    results.append({"gen_id": gen_id, "status": "ok"})
                except Exception as e:
                    results.append({"gen_id": gen_id, "status": "error", "error": str(e)})
                if (i + 1) % 20 == 0:
                    logger.info(f"[InferActor] {i+1}/{len(samples)} done")
            return results

    return InferActor


# ─── Ray Actor: WER Eval ─────────────────────────────────────
def create_wer_actor_class():
    import ray

    @ray.remote(num_gpus=1)
    class WERActor:
        def __init__(self):
            import string
            import torch
            import soundfile as sf
            import scipy.signal
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            self.sf = sf
            self.scipy_signal = scipy.signal
            self.torch = torch
            self.punctuation_all = string.punctuation

            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3", torch_dtype=torch.float32
            ).to("cuda")
            self.asr_model.eval()
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            logger.info("[WERActor] transformers Whisper large-v3 loaded (float32)")

        def _normalize_en(self, text):
            for x in self.punctuation_all:
                if x == "'":
                    continue
                text = text.replace(x, "")
            text = text.replace("  ", " ")
            return text.lower()

        def eval_batch(self, samples):
            """samples: list of (gen_wav, gen_text)"""
            import jiwer
            results = []
            for gen_wav, truth in samples:
                try:
                    wav, sr = self.sf.read(gen_wav)
                    if sr != 16000:
                        wav = self.scipy_signal.resample(wav, int(len(wav) * 16000 / sr))
                    input_features = self.processor(
                        wav, sampling_rate=16000, return_tensors="pt"
                    ).input_features.to("cuda")
                    with self.torch.no_grad():
                        predicted_ids = self.asr_model.generate(
                            input_features, forced_decoder_ids=self.forced_decoder_ids
                        )
                    hypo_raw = self.processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]
                except Exception as e:
                    results.append({"wav": Path(gen_wav).stem, "wer": None, "error": str(e)})
                    continue

                truth_norm = self._normalize_en(truth)
                hypo_norm = self._normalize_en(hypo_raw)

                if not truth_norm.strip():
                    results.append({"wav": Path(gen_wav).stem, "wer": None, "error": "empty_ref"})
                    continue

                wer_val = float(jiwer.wer(truth_norm, hypo_norm))
                results.append({
                    "wav": Path(gen_wav).stem,
                    "truth": truth,
                    "hypo": hypo_raw,
                    "wer": wer_val,
                })
            return results

    return WERActor


# ─── Ray Actor: WER Eval (CV3-Eval official: openai-whisper) ─
def create_cv3_wer_actor_class():
    """CV3-Eval official: openai-whisper large-v3, model.transcribe(language='en').
    Matches https://github.com/FunAudioLLM/CV3-Eval/blob/main/utils/run_wer.py"""
    import ray

    @ray.remote(num_gpus=1)
    class CV3WERActor:
        def __init__(self):
            import string
            import whisper
            import soundfile as sf
            self.sf = sf
            self.model = whisper.load_model("large-v3", device="cuda")
            self.model.eval()
            self.punctuation_all = string.punctuation
            logger.info("[CV3WERActor] openai-whisper large-v3 loaded (CV3-Eval official)")

        def _normalize_en(self, text):
            for x in self.punctuation_all:
                if x == "'":
                    continue
                text = text.replace(x, '')
            text = text.replace('  ', ' ')
            return text.lower()

        def eval_batch(self, samples):
            """samples: list of (gen_wav, gen_text)"""
            import jiwer
            results = []
            for gen_wav, truth in samples:
                try:
                    result = self.model.transcribe(gen_wav, language="en")
                    hypo_raw = result["text"].strip()
                except Exception as e:
                    results.append({"wav": Path(gen_wav).stem, "wer": None, "error": str(e)})
                    continue

                truth_norm = self._normalize_en(truth)
                hypo_norm = self._normalize_en(hypo_raw)

                if not truth_norm.strip():
                    results.append({"wav": Path(gen_wav).stem, "wer": None, "error": "empty_ref"})
                    continue

                wer_val = float(jiwer.wer(truth_norm, hypo_norm))
                results.append({
                    "wav": Path(gen_wav).stem,
                    "truth": truth,
                    "hypo": hypo_raw,
                    "wer": wer_val,
                })
            return results

    return CV3WERActor


# ─── Ray Actor: SIM Eval ─────────────────────────────────────
def create_sim_actor_class():
    import ray

    @ray.remote(num_gpus=1)
    class SIMActor:
        def __init__(self, wavlm_ckpt: str, s3prl_cache: str):
            import torch
            import torchaudio
            import torch.nn.functional as F
            from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

            self.torch = torch
            self.torchaudio = torchaudio
            self.F = F
            self.device = "cuda:0"

            model = ECAPA_TDNN_SMALL(
                feat_dim=1024, feat_type="wavlm_large", config_path=None
            )
            state_dict = torch.load(wavlm_ckpt, weights_only=True, map_location="cpu")
            model.load_state_dict(state_dict["model"], strict=False)
            model = model.cuda(self.device)
            model.eval()
            self.model = model
            logger.info("[SIMActor] ECAPA-TDNN + WavLM loaded")

        def eval_batch(self, samples):
            """samples: list of (gen_wav, ref_wav)"""
            results = []
            for gen_wav, ref_wav in samples:
                try:
                    wav1, sr1 = self.torchaudio.load(gen_wav)
                    wav2, sr2 = self.torchaudio.load(ref_wav)

                    if sr1 != 16000:
                        wav1 = self.torchaudio.functional.resample(wav1, sr1, 16000)
                    if sr2 != 16000:
                        wav2 = self.torchaudio.functional.resample(wav2, sr2, 16000)

                    wav1 = wav1.cuda(self.device)
                    wav2 = wav2.cuda(self.device)

                    with self.torch.no_grad():
                        emb1 = self.model(wav1)
                        emb2 = self.model(wav2)

                    sim = self.F.cosine_similarity(emb1, emb2)[0].item()
                    results.append({"wav": Path(gen_wav).stem, "sim": sim})
                except Exception as e:
                    results.append({"wav": Path(gen_wav).stem, "sim": None, "error": str(e)})
            return results

    return SIMActor


# ─── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TTS Inference + Eval (WER + SIM) with Ray")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path (.pt)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Model config yaml (resolved Hydra config.yaml or raw config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: auto)")
    parser.add_argument("--meta_lst", type=str, default=META_LST, help="seed-tts-eval meta.lst path")
    parser.add_argument("--wavlm_ckpt", type=str, default=WAVLM_CKPT, help="WavLM finetuned checkpoint")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--skip_infer", action="store_true", help="Skip inference (eval only)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation (infer only)")
    args = parser.parse_args()

    # Auto output dir from checkpoint name
    if args.output_dir is None:
        ckpt_name = Path(args.ckpt).stem
        ckpt_parent = Path(args.ckpt).parent.name
        args.output_dir = str(PROJ_ROOT / "eval_outputs" / f"{ckpt_parent}_{ckpt_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    logger.info(f"[config]  {args.config}")
    logger.info(f"[ckpt]    {args.ckpt}")
    logger.info(f"[output]  {args.output_dir}")
    logger.info(f"[gpus]    {args.num_gpus}")

    # Parse meta.lst
    samples = parse_meta_lst(args.meta_lst)
    logger.info(f"[meta]    {len(samples)} samples from {args.meta_lst}")

    # ── Init Ray ──
    import ray
    ray.init(
        num_gpus=args.num_gpus,
        ignore_reinit_error=True,
    )

    # ══════════════════════════════════════════════════════════
    # Phase 1: Inference
    # ══════════════════════════════════════════════════════════
    if not args.skip_infer:
        logger.info("=" * 60)
        logger.info("Phase 1: INFERENCE")
        logger.info("=" * 60)

        InferActor = create_infer_actor_class()

        # Split samples across GPUs
        chunks = [[] for _ in range(args.num_gpus)]
        for i, s in enumerate(samples):
            chunks[i % args.num_gpus].append(s)

        actors = [InferActor.remote(args.config, args.ckpt) for _ in range(args.num_gpus)]
        futures = [a.infer_batch.remote(chunk, wav_dir) for a, chunk in zip(actors, chunks)]

        t0 = time.time()
        all_results = ray.get(futures)
        elapsed = time.time() - t0

        total = sum(len(r) for r in all_results)
        errors = sum(1 for r in all_results for x in r if x["status"] == "error")
        logger.info(f"[Inference] {total} samples in {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/sample)")
        if errors:
            logger.warning(f"[Inference] {errors} errors")

        # Save inference log
        flat = [x for r in all_results for x in r]
        with open(os.path.join(args.output_dir, "infer_log.json"), "w") as f:
            json.dump(flat, f, indent=2)

        # Cleanup inference actors
        for a in actors:
            ray.kill(a)
        del actors
        import gc; gc.collect()
        time.sleep(2)

    # ══════════════════════════════════════════════════════════
    # Phase 2: Evaluation
    # ══════════════════════════════════════════════════════════
    if not args.skip_eval:
        logger.info("=" * 60)
        logger.info("Phase 2: EVALUATION (WER + SIM)")
        logger.info("=" * 60)

        # Build eval sample lists
        wer_samples = []  # (gen_wav, gen_text)
        sim_samples = []  # (gen_wav, ref_wav)

        for gen_id, ref_text, ref_audio, gen_text in samples:
            gen_wav = os.path.join(wav_dir, f"{gen_id}.wav")
            if not os.path.exists(gen_wav):
                continue
            wer_samples.append((gen_wav, gen_text))
            sim_samples.append((gen_wav, ref_audio))

        logger.info(f"[Eval] {len(wer_samples)} samples with generated wavs")

        # GPU allocation: with >=2 GPUs split WER/SIM in parallel;
        # with 1 GPU run WER then SIM sequentially (actors need full GPU lifetime)
        single_gpu = args.num_gpus == 1
        n_wer_gpus = 1 if single_gpu else max(1, args.num_gpus // 2)
        n_sim_gpus = 1 if single_gpu else max(1, args.num_gpus - n_wer_gpus)

        # cv3/cv3-hard: use openai-whisper (CV3-Eval official)
        # others: use transformers Whisper (seed-tts-eval official)
        _is_cv3 = "cv3" in os.path.basename(args.meta_lst).lower()
        if _is_cv3:
            logger.info("[WER] Using openai-whisper (CV3-Eval official)")
            WERActor = create_cv3_wer_actor_class()
        else:
            WERActor = create_wer_actor_class()
        SIMActor = create_sim_actor_class()

        # Pre-download s3prl to avoid race condition between actors
        s3prl_cache = os.path.expanduser("~/.cache/torch/hub/s3prl_s3prl_main")
        if not os.path.isdir(s3prl_cache):
            logger.info("[SIM] Pre-downloading s3prl cache...")
            import torch as _torch
            _torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            _torch.hub.load("s3prl/s3prl", "wavlm_large")
            logger.info("[SIM] s3prl cache ready")

        t0 = time.time()

        # ── WER ──
        wer_chunks = [[] for _ in range(n_wer_gpus)]
        for i, s in enumerate(wer_samples):
            wer_chunks[i % n_wer_gpus].append(s)

        wer_actors = [WERActor.remote() for _ in range(n_wer_gpus)]

        if single_gpu:
            # Sequential: WER first, then kill actors before SIM
            wer_futures = [a.eval_batch.remote(c) for a, c in zip(wer_actors, wer_chunks)]
            wer_results_all = ray.get(wer_futures)
            for a in wer_actors:
                ray.kill(a)
            del wer_actors
            import gc; gc.collect()
        else:
            # Parallel: dispatch both WER and SIM simultaneously
            wer_futures = [a.eval_batch.remote(c) for a, c in zip(wer_actors, wer_chunks)]

        # ── SIM ──
        sim_chunks = [[] for _ in range(n_sim_gpus)]
        for i, s in enumerate(sim_samples):
            sim_chunks[i % n_sim_gpus].append(s)

        sim_actors = [SIMActor.remote(args.wavlm_ckpt, s3prl_cache) for _ in range(n_sim_gpus)]
        sim_futures = [a.eval_batch.remote(c) for a, c in zip(sim_actors, sim_chunks)]

        # ── Gather results ──
        if not single_gpu:
            wer_results_all = ray.get(wer_futures)

        wer_flat = [x for r in wer_results_all for x in r]
        wer_values = [x["wer"] for x in wer_flat if x.get("wer") is not None]
        wer_mean = float(np.mean(wer_values)) if wer_values else None

        sim_results_all = ray.get(sim_futures)
        sim_flat = [x for r in sim_results_all for x in r]
        sim_values = [x["sim"] for x in sim_flat if x.get("sim") is not None]
        sim_mean = float(np.mean(sim_values)) if sim_values else None

        elapsed = time.time() - t0

        # ── Results ──
        logger.info("=" * 60)
        logger.info(f"RESULTS ({len(wer_values)} samples, {elapsed:.1f}s)")
        logger.info("=" * 60)
        logger.info(f"  WER:  {wer_mean:.4f}" if wer_mean is not None else "  WER:  N/A")
        logger.info(f"  SIM:  {sim_mean:.4f}" if sim_mean is not None else "  SIM:  N/A")
        logger.info("=" * 60)

        # Save results
        summary = {
            "ckpt": args.ckpt,
            "config": args.config,
            "num_samples": len(wer_values),
            "wer_mean": round(wer_mean, 5) if wer_mean else None,
            "sim_mean": round(sim_mean, 5) if sim_mean else None,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(args.output_dir, "wer_results.jsonl"), "w") as f:
            for x in wer_flat:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        with open(os.path.join(args.output_dir, "sim_results.jsonl"), "w") as f:
            for x in sim_flat:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        logger.info(f"[Saved] {args.output_dir}/summary.json")
        logger.info(f"[Saved] {args.output_dir}/wer_results.jsonl")
        logger.info(f"[Saved] {args.output_dir}/sim_results.jsonl")

    ray.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    main()
