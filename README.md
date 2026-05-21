<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-OpenTTS-Gradient-White.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/Raon-OpenTTS-Gradient-Black.png">
    <img alt="Raon-OpenTTS" src="assets/Raon-OpenTTS-Gradient-Black.png" width="600">
  </picture>
</div>

# Raon-OpenTTS

**Open Models and Data for Robust Text-to-Speech**

[![arXiv](https://img.shields.io/badge/arXiv-2605.20830-red?style=flat)](https://arxiv.org/abs/2605.20830)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KRAFTON-yellow?style=flat)](https://huggingface.co/KRAFTON)
[![Model 0.3B](https://img.shields.io/badge/Model-Raon--OpenTTS--0.3B-blue?style=flat)](https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B)
[![Model 1B](https://img.shields.io/badge/Model-Raon--OpenTTS--1B-blue?style=flat)](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B)
[![Dataset](https://img.shields.io/badge/Dataset-Raon--OpenTTS--Pool-blue?style=flat)](https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool)
[![Eval](https://img.shields.io/badge/Eval-Raon--OpenTTS--Eval-blue?style=flat)](https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Eval)

**[Technical Report](https://arxiv.org/abs/2605.20830)** | **[Raon-OpenTTS-1B](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B)**

## Highlights

- **Fully open**: both model weights and training data are publicly available.
- **Large-scale training**: 510.1K hours of quality-filtered speech (Raon-OpenTTS-Core), drawn from a 615K-hour open pool (Raon-OpenTTS-Pool) comprising 11 English datasets.
- **Competitive with closed-data SOTA**: matches or outperforms MaskGCT, VoxCPM, CosyVoice 3, and Qwen3-TTS on standard benchmarks while being the first system that is simultaneously open-weight and open-data at this scale.
- **Two model sizes**: 0.3B and 1B parameters, both based on the F5-TTS DiT architecture.

## Model Zoo

| Model | Params | Architecture | Training Data | Download |
|-------|--------|-------------|---------------|----------|
| Raon-OpenTTS-0.3B | 336M | DiT (dim=1024, depth=22, heads=16, ff_mult=2) | Raon-OpenTTS-Core (510.1K hrs) | [HuggingFace](https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B) |
| Raon-OpenTTS-1B | 1048M | DiT (dim=1408, depth=28, heads=24, ff_mult=4) | Raon-OpenTTS-Core (510.1K hrs) | [HuggingFace](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B) |

Both models use character-level tokenization (vocab size 5,512) with text_dim=512, and are trained on 80-channel log mel-spectrograms at 16 kHz (hop=256). A pretrained HiFi-GAN vocoder (16 kHz, LibriTTS) is used for waveform synthesis.

## Benchmark Results

Bold marks the best result and the Raon-OpenTTS rows. All numbers are from the paper ([arXiv:2605.20830](https://arxiv.org/abs/2605.20830)).

### Seed-TTS-Eval (English)

WER measured via Whisper-large-v3; speaker similarity (SIM) via WavLM-large.

| Model | Params | Training Data | Open-Weight | Open-Data | WER (%) ↓ | SIM ↑ |
|-------|--------|---------------|:-----------:|:---------:|--------:|----:|
| Human | - | - | - | - | 2.14 | 0.734 |
| Seed-TTS | - | - | | | 2.25 | 0.762 |
| CosyVoice 3 | 1.5B | ~1M hrs | | | 2.21 | 0.720 |
| Index-TTS 2 | 1.5B | 55K hrs | Yes | | 2.18 | 0.709 |
| Llasa | 8B | 250K hrs | Yes | | 3.63 | 0.581 |
| VoxCPM | 0.5B | 1.8M hrs | Yes | | 1.98 | 0.730 |
| CosyVoice 2 | 0.5B | 170K hrs | Yes | | 2.61 | 0.659 |
| CosyVoice 3 | 0.5B | ~1M hrs | Yes | | 2.50 | 0.698 |
| Qwen3-TTS | 1.7B | ~5M hrs | Yes | | **1.46** | 0.715 |
| Voxtral TTS | 4B | - | Yes | | 2.19 | 0.663 |
| MaskGCT | 0.6B | 100K hrs | Yes | Yes | 2.57 | 0.713 |
| F5-TTS | 0.3B | 100K hrs | Yes | Yes | 2.04 | 0.671 |
| **Raon-OpenTTS-0.3B** | 0.3B | 510K hrs | Yes | Yes | 1.95 | 0.687 |
| **Raon-OpenTTS-1B** | 1.0B | 510K hrs | Yes | Yes | 1.78 | **0.749** |

### CV3-Eval

WER on CV3-EN and CV3-Hard-EN; SIM via ERes2Net, DNSMOS for perceptual quality (CV3-Hard-EN).

| Model | CV3-EN WER (%) ↓ | CV3-Hard-EN WER (%) ↓ | CV3-Hard-EN SIM ↑ | CV3-Hard-EN DNSMOS ↑ |
|-------|---------------:|---------------------:|----------------:|-------------------:|
| F5-TTS | 8.54 | - | - | - |
| MaskGCT | 7.73 | 41.09 | 0.624 | 3.48 |
| CosyVoice 2 | 6.27 | 10.28 | 0.710 | 3.95 |
| CosyVoice 3 | 4.96 | 10.77 | 0.740 | **3.98** |
| VoxCPM | 5.24 | 6.44 | 0.670 | 3.78 |
| Qwen3-TTS | 4.52 | 7.89 | 0.666 | 3.87 |
| **Raon-OpenTTS-0.3B** | 4.62 | 7.31 | 0.730 | 3.77 |
| **Raon-OpenTTS-1B** | **3.92** | **6.15** | **0.775** | 3.85 |

### Raon-OpenTTS-Eval

Covers 4 acoustic regimes (Clean, Noisy, Wild, Expressive) across 12 datasets with 6K prompt-text pairs. Overall is computed over all evaluation samples.

| Model | Clean WER ↓ | Clean SIM ↑ | Noisy WER ↓ | Noisy SIM ↑ | Wild WER ↓ | Wild SIM ↑ | Expr. WER ↓ | Expr. SIM ↑ | Overall WER ↓ | Overall SIM ↑ |
|-------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| F5-TTS | 2.17 | 0.613 | 3.82 | 0.640 | 136.03 | 0.324 | 3.46 | 0.503 | 25.08 | 0.542 |
| MaskGCT | 3.39 | 0.672 | 5.56 | 0.727 | 28.00 | 0.581 | 6.44 | 0.546 | 8.61 | 0.635 |
| CosyVoice 2 | 2.59 | 0.642 | 4.39 | 0.675 | 49.73 | 0.535 | 3.66 | 0.536 | 11.02 | 0.603 |
| CosyVoice 3 | 2.53 | 0.678 | 3.69 | 0.720 | 8.31 | 0.618 | 5.49 | 0.567 | 4.43 | 0.647 |
| VoxCPM | 2.24 | 0.686 | **3.42** | 0.738 | 43.83 | 0.553 | 2.66 | 0.565 | 9.48 | 0.642 |
| Qwen3-TTS | 3.38 | 0.684 | 4.60 | 0.726 | 79.14 | 0.528 | 5.81 | 0.527 | 17.59 | 0.626 |
| **Raon-OpenTTS-0.3B** | 1.57 | 0.645 | 4.03 | 0.700 | 5.83 | 0.571 | **2.53** | 0.570 | 2.93 | 0.623 |
| **Raon-OpenTTS-1B** | **1.44** | **0.718** | 3.51 | **0.769** | **5.61** | **0.656** | 2.77 | **0.633** | **2.81** | **0.695** |

## Installation

```bash
git clone https://github.com/krafton-ai/Raon-OpenTTS.git
cd Raon-OpenTTS
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
    --ckpt_dir checkpoints/Raon-OpenTTS-0.3B \
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

Both models are trained from the [Raon-OpenTTS-Pool](https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool) HuggingFace dataset using the `core` split (quality-filtered). 

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
| Raon-OpenTTS-Eval | WER, SIM | 4 acoustic regimes (Clean, Noisy, Wild, Expressive), 12 datasets, 6K prompt-text pairs |

```bash
# Run evaluation across all benchmarks
bash src/f5_tts/eval/run_infer_eval.sh
```

## Data

**Raon-OpenTTS-Pool** (615K hours, 11 English speech datasets) is publicly available on [HuggingFace](https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool)

**Raon-OpenTTS-Core** (510.1K hours, 194.5M segments) is the quality-filtered subset used for training. It is obtained by applying a combined filter based on DNSMOS, WER, and VAD rank scores, removing the bottom 15% of Raon-OpenTTS-Pool. The `core` split in the HuggingFace dataset corresponds to Raon-OpenTTS-Core.

## Acknowledgement

This project is built upon [F5-TTS](https://github.com/SWivid/F5-TTS) by SWivid. We thank the authors for their excellent open-source work.

## License

This project is licensed under [Apache 2.0](LICENSE).

## Citation

```bibtex
@article{kim2026raonopentts,
    title={Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech},
    author={Kim, Semin and Chung, Seungjun and Moon, Taehong and Lee, Sangheon and Ahn, Minyoung and Lee, Keon and Kim, Nam Soo and Cho, Jaewoong and Schmidt, Ludwig and Lee, Kangwook and Park, Dongmin},
    journal={arXiv preprint arXiv:2605.20830},
    year={2026}
}
```
