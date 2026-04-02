<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Raon-OpenTTS-Gradient-White.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/Raon-OpenTTS-Gradient-Black.png">
    <img alt="Raon-OpenTTS" src="assets/Raon-OpenTTS-Gradient-Black.png" width="600">
  </picture>
</div>

# Raon-OpenTTS

**Open Models and Data for Robust Text-to-Speech**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![Model 0.3B](https://img.shields.io/badge/HuggingFace-RAON--TTS--0.3B-yellow)](https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B)
[![Model 1B](https://img.shields.io/badge/HuggingFace-RAON--TTS--1B-yellow)](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B)
[![Dataset](https://img.shields.io/badge/HuggingFace-RAON--TTS--Pool-blue)](https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool)

## Highlights

- **Fully open**: both model weights and training data are publicly available.
- **Large-scale training**: 510.1K hours of quality-filtered speech (Raon-OpenTTS-Core), drawn from a 615K-hour open pool (Raon-OpenTTS-Pool) comprising 11 English datasets.
- **Competitive with closed-data SOTA**: matches or outperforms MaskGCT, VoxCPM, CosyVoice 3, and Qwen3-TTS on standard benchmarks while being the first system that is simultaneously open-weight and open-data at this scale.
- **Two model sizes**: 0.3B and 1B parameters, both based on the F5-TTS DiT architecture.

## Model Zoo

| Model | Params | Architecture | Training Data | Download |
|-------|--------|-------------|---------------|----------|
| Raon-OpenTTS-0.3B | 336M | DiT (dim=1024, depth=22, heads=16, ff_mult=2) | Raon-OpenTTS-Core (510.1K hrs) | [HuggingFace](https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B) |
| Raon-OpenTTS-1B | 1048M | DiT (dim=1408, depth=28, heads=24, ff_mult=4) | Raon-OpenTTS-Core (510.1K hrs) | TBD |

Both models use character-level tokenization (vocab size 5,512) with text_dim=512, and are trained on 80-channel log mel-spectrograms at 16 kHz (hop=256). A pretrained HiFi-GAN vocoder (16 kHz, LibriTTS) is used for waveform synthesis.

## Benchmark Results

### Seed-TTS-Eval (English)

WER measured via Whisper-large-v3; speaker similarity (SIM) via WavLM-large.

TBD

### CV3-Eval

SIM measured via ERes2Net.

TBD

### Raon-OpenTTS-Eval

Covers 4 acoustic regimes (Clean, Noisy, Wild, Emotional) across 12 datasets with 6K prompt-text pairs.

TBD

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
| Raon-OpenTTS-Eval | WER, SIM | 4 acoustic regimes (Clean, Noisy, Wild, Emotional), 12 datasets, 6K prompt-text pairs |

```bash
# Run evaluation across all benchmarks
bash src/f5_tts/eval/run_infer_eval.sh
```

## Data

**Raon-OpenTTS-Pool** (615K hours, 11 English speech datasets) is publicly available on HuggingFace:
[https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool](https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool)

**Raon-OpenTTS-Core** (510.1K hours, 194.5M segments) is the quality-filtered subset used for training. It is obtained by applying a combined filter based on DNSMOS, WER, and VAD rank scores, removing the bottom 15% of Raon-OpenTTS-Pool. The `core` split in the HuggingFace dataset corresponds to Raon-OpenTTS-Core.

## Acknowledgement

This project is built upon [F5-TTS](https://github.com/SWivid/F5-TTS) by SWivid. We thank the authors for their excellent open-source work.

## License

This project is licensed under [Apache 2.0](LICENSE).

## Citation

```bibtex
@article{raonopentts2026,
    title={Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech},
    author={},
    year={2026}
}
```
