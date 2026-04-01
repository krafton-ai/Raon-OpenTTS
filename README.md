<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-OpenTTS-Gradient-White.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/Raon-OpenTTS-Gradient-Black.png">
    <img alt="RAON-OpenTTS" src="assets/Raon-OpenTTS-Gradient-Black.png" width="600">
  </picture>
</div>

# RAON-OpenTTS

**Open Models and Data for Robust Text-to-Speech**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![Model 0.3B](https://img.shields.io/badge/HuggingFace-RAON--TTS--0.3B-yellow)](https://huggingface.co/KRAFTON/RAON-TTS-0.3B)
[![Model 1B](https://img.shields.io/badge/HuggingFace-RAON--TTS--1B-yellow)](https://huggingface.co/KRAFTON/RAON-TTS-1B)
[![Dataset](https://img.shields.io/badge/HuggingFace-RAON--TTS--Pool-blue)](https://huggingface.co/datasets/KRAFTON/RAON-TTS-Pool)

## Highlights

- **Fully open**: both model weights and training data are publicly available.
- **Large-scale training**: 510.1K hours of quality-filtered speech (RAON-TTS-Core), drawn from a 615K-hour open pool (RAON-TTS-Pool) comprising 11 English datasets.
- **Competitive with closed-data SOTA**: matches or outperforms MaskGCT, VoxCPM, CosyVoice 3, and Qwen3-TTS on standard benchmarks while being the first system that is simultaneously open-weight and open-data at this scale.
- **Two model sizes**: 0.3B and 1B parameters, both based on the F5-TTS DiT architecture.

## Model Zoo

| Model | Params | Architecture | Training Data | Download |
|-------|--------|-------------|---------------|----------|
| RAON-TTS-0.3B | 336M | DiT (dim=1024, depth=22, heads=16, ff_mult=2) | RAON-TTS-Core (510.1K hrs) | [HuggingFace](https://huggingface.co/KRAFTON/RAON-TTS-0.3B) |
| RAON-TTS-1B | 1048M | DiT (dim=1408, depth=28, heads=24, ff_mult=4) | RAON-TTS-Core (510.1K hrs) | [HuggingFace](https://huggingface.co/KRAFTON/RAON-TTS-1B) |

Both models use character-level tokenization (vocab size 5,512) with text_dim=512, and are trained on 80-channel log mel-spectrograms at 16 kHz (hop=256). A pretrained HiFi-GAN vocoder (16 kHz, LibriTTS) is used for waveform synthesis.

## Benchmark Results

### Seed-TTS-Eval (English)

WER measured via Whisper-large-v3; speaker similarity (SIM) via WavLM-large.

| Model | Params | Training Data | Open-Weight | Open-Data | WER (%) | SIM |
|-------|--------|---------------|:-----------:|:---------:|--------:|----:|
| Human | - | - | - | - | 2.14 | 0.734 |
| Seed-TTS | - | - | | | 2.25 | 0.762 |
| CosyVoice 3 | 1.5B | ~1M hrs | | | 2.21 | 0.720 |
| Qwen3-TTS | 1.7B | ~5M hrs | Yes | | 1.46 | 0.715 |
| F5-TTS (32 NFE) | 0.3B | 100K hrs | Yes | Yes | 2.04 | 0.671 |
| **RAON-TTS (0.3B)** | 0.3B | 510K hrs | Yes | Yes | 1.97 | 0.703 |
| **RAON-TTS (1B)** | 1.0B | 510K hrs | Yes | Yes | **1.91** | **0.737** |

### CV3-Eval

SIM measured via ERes2Net.

| Model | CV3-EN WER (%) | CV3-Hard-EN WER (%) | CV3-Hard-EN SIM | CV3-Hard-EN DNSMOS |
|-------|---------------:|---------------------:|----------------:|-------------------:|
| F5-TTS | 7.43 | 12.06 | 0.683 | 3.81 |
| CosyVoice 3 | 5.44 | 9.61 | 0.742 | 3.95 |
| Qwen3-TTS | 4.28 | 5.93 | 0.665 | 3.89 |
| **RAON-TTS (1B)** | **3.97** | 6.49 | **0.766** | 3.91 |

### RAON-TTS-Eval

Covers 4 acoustic regimes (Clean, Noisy, Wild, Emotional) across 12 datasets with 6K prompt-text pairs.

| Model | Clean WER (%) | Noisy WER (%) | Wild WER (%) | Emo WER (%) | Overall WER (%) | Overall SIM |
|-------|-------------:|-------------:|------------:|------------:|----------------:|------------:|
| F5-TTS | 1.50 | 7.70 | 77.50 | 3.70 | 16.20 | 0.589 |
| CosyVoice 3 | 2.29 | 3.70 | 10.60 | 5.40 | 4.68 | 0.658 |
| Qwen3-TTS | 2.83 | 3.60 | 37.90 | 2.02 | 8.09 | 0.655 |
| **RAON-TTS (1B)** | **1.30** | 3.76 | **7.17** | **2.14** | **2.90** | **0.667** |

## Installation

```bash
git clone https://github.com/krafton-ai/RAON-OpenTTS.git
cd RAON-OpenTTS
pip install -e .

# With evaluation dependencies (WER, SIM, DNSMOS)
pip install -e ".[eval]"
```

### Vocoder

We use a HiFi-GAN vocoder fine-tuned on LibriTTS at 16 kHz (originally from [speechbrain/tts-hifigan-libritts-16kHz](https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz)). Our standalone loader requires no speechbrain dependency.

```bash
mkdir -p pretrained_models
huggingface-cli download speechbrain/tts-hifigan-libritts-16kHz generator.ckpt --local-dir pretrained_models
```

## Quick Start: Inference

```bash
python -m f5_tts.infer.infer_cli \
    --config src/f5_tts/configs/03b.yaml \
    --ckpt_dir checkpoints/RAON-TTS-0.3B \
    --ckpt_name model_last.pt \
    --lst_path data/librispeech_pc_test_clean_cross_sentence.lst \
    --audio_root data/librispeech/test-clean \
    --output_dir output/inference
```

### VAD-based Duration Estimation

The inference pipeline uses VAD-trimmed reference length for generation-length estimation, while the original (non-trimmed) audio is used as the conditioning signal. A dynamic silence threshold adapts to the speaker's volume, and a minimum speech rate (12 chars/sec) is enforced to prevent excessively long generations.

```python
from f5_tts.infer.utils_infer import infer_process

audio, sr, _ = infer_process(ref_audio, ref_text, gen_text, model, vocoder)
```

## Training

Both models are trained from the [RAON-TTS-Pool](https://huggingface.co/datasets/KRAFTON/RAON-TTS-Pool) HuggingFace dataset using the `core` split (quality-filtered). 

### Launch training

```bash
# 0.3B model (1 nodes x 8 GPUs)
accelerate launch --multi_gpu --mixed_precision bf16 \
    --num_processes 8 --num_machines 1 \
    -m f5_tts.train.train --config-name=03b

# 1B model (1 nodes x 8 GPUs)
accelerate launch --multi_gpu --mixed_precision bf16 \
    --num_processes 8 --num_machines 1 \
    -m f5_tts.train.train --config-name=1b
```

### Adapting to different hardware

If you use a different number of GPUs or batch size, recompute `total_updates_per_epoch` with a dry run:

```bash
# Run one step and check log output for "TOTAL UPDATES <N>"
accelerate launch --multi_gpu --mixed_precision bf16 \
    --num_processes <num_gpus> \
    -m f5_tts.train.train --config-name=03b
# Then set: total_updates_per_epoch = TOTAL_UPDATES / (epochs x num_gpus)
```

## Evaluation

We evaluate on 3 benchmarks measuring intelligibility (WER) and speaker similarity (SIM):

| Benchmark | Metrics | Description |
|-----------|---------|-------------|
| Seed-TTS-Eval (EN) | WER (Whisper-large-v3), SIM (WavLM-large) | Standard zero-shot TTS evaluation with cross-sentence prompts |
| CV3-Eval | WER, SIM (ERes2Net), DNSMOS | CV3-EN and CV3-Hard-EN subsets with diverse speakers |
| RAON-TTS-Eval | WER, SIM | 4 acoustic regimes (Clean, Noisy, Wild, Emotional), 12 datasets, 6K prompt-text pairs |

```bash
# Run evaluation across all benchmarks
bash src/f5_tts/eval/run_infer_eval.sh
```

## Data

**RAON-TTS-Pool** (615K hours, 11 English speech datasets) is publicly available on HuggingFace:
[https://huggingface.co/datasets/KRAFTON/RAON-TTS-Pool](https://huggingface.co/datasets/KRAFTON/RAON-TTS-Pool)

**RAON-TTS-Core** (510.1K hours, 194.5M segments) is the quality-filtered subset used for training. It is obtained by applying a combined filter based on DNSMOS, WER, and VAD rank scores, removing the bottom 15% of RAON-TTS-Pool. The `core` split in the HuggingFace dataset corresponds to RAON-TTS-Core.

## Acknowledgement

This project is built upon [F5-TTS](https://github.com/SWivid/F5-TTS) by SWivid. We thank the authors for their excellent open-source work.

## License

This project is licensed under [Apache 2.0](LICENSE).

## Citation

```bibtex
@article{raonopentts2026,
    title={RAON-OpenTTS: Open Models and Data for Robust Text-to-Speech},
    author={},
    year={2026}
}
```
