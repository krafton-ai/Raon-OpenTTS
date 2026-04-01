import sys, os
import argparse
import logging
import torch
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
import random
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
# -----------------------------
# argparse
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--ckpt_name", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--lst_path", type=str, required=True, help="Path to LST file")
parser.add_argument("--audio_root", type=str, required=True, help="Root directory for reference audio")
parser.add_argument("--max_samples", type=int, default=500, help="Number of samples to randomly select from LST")
parser.add_argument(
    "--vocab_path",
    type=str,
    default=None,
    help="Path to Emilia-style vocab.txt (one token per line). Falls back to default tokenizer if not set",
)

args = parser.parse_args()

# -----------------------------
# Default settings
# -----------------------------

from ..model import CFM
from .utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    sway_sampling_coef,
    target_rms,
    speed,
)
from ..model.utils import get_tokenizer
from ema_pytorch import EMA


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

# -----------------------------
# Load config
# -----------------------------
model_cfg = OmegaConf.load(args.config)

model_cls = hydra.utils.get_class(
    f"f5_tts.model.{model_cfg.model.backbone}"
)
model_arc = model_cfg.model.arch
tokenizer_name = model_cfg.model.tokenizer
mel_spec_cfg = model_cfg.model.mel_spec

if args.vocab_path is not None:
    # 1) Use Emilia-style vocab.txt (unified script approach)
    vocab_file = args.vocab_path
    tokenizer_name = "custom"  # Emilia vocab is treated as custom tokenizer

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer_name)

    logger.info(f"> Using Emilia vocab from {vocab_file}")
    logger.info(f"> Vocab size: {vocab_size}")

else:
    # 2) Use original F5 tokenizer logic as-is
    tokenizer_name = model_cfg.model.tokenizer

    tokenizer_path = model_cfg.model.tokenizer_path

    paths = tokenizer_path.split("|")
    char_maps = [get_tokenizer(p, tokenizer_name)[0] for p in paths]
    all_tokens = {tok for cmap in char_maps for tok in cmap.keys()}
    vocab_char_map = {tok: idx for idx, tok in enumerate(sorted(all_tokens))}
    vocab_size = len(vocab_char_map)

    logger.info(f"> Using original tokenizer: {tokenizer_name}")
    logger.info(f"> Vocab size: {vocab_size}")

# -----------------------------
# Create model
# -----------------------------
model = CFM(
    transformer=model_cls(
        **model_arc,
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_cfg.n_mel_channels,
    ),
    mel_spec_kwargs=mel_spec_cfg,
    vocab_char_map=vocab_char_map,
).to(device)

ema = EMA(model, include_online_model=False).to(device)

# -----------------------------
# Load checkpoint
# -----------------------------
ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
logger.info(f"> Loading checkpoint: {ckpt_path}")

if ckpt_path.endswith(".safetensors"):
    # safetensors only supports CPU loading
    ema_state = load_file(ckpt_path)  # no device arg (default = cpu)
    ckpt = {"ema_model_state_dict": ema_state}
else:
    # .pt, .pth also loaded safely on CPU
    ckpt = torch.load(ckpt_path, map_location="cpu")

if "ema_model_state_dict" not in ckpt:
    raise RuntimeError("Checkpoint does not contain 'ema_model_state_dict'.")

# ema is already moved to device via .to(device)
ema.load_state_dict(ckpt["ema_model_state_dict"])

for key, param in ema.ema_model.state_dict().items():
    model.state_dict()[key].copy_(param)

model.eval()
logger.info("> EMA weights applied to CFM.")
# Load vocoder
# -----------------------------
vocoder = load_vocoder(
    vocoder_name=model_cfg.model.mel_spec.mel_spec_type,
    is_local=model_cfg.model.vocoder.is_local,
    local_path=model_cfg.model.vocoder.local_path,
)

try:
    target_sample_rate = model.mel_spec.target_sample_rate
except AttributeError:
    target_sample_rate = 22050

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Read LST file
# -----------------------------
with open(args.lst_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
if args.max_samples is not None and args.max_samples > 0:
    rng = random.Random(42)
    rng.shuffle(lines)
    lines = lines[: args.max_samples]
logger.info(f"Loaded {len(lines)} lines from {args.lst_path}")

# -----------------------------
# Inference
# -----------------------------
for i, line in enumerate(tqdm(lines)):
    fields = line.split("\t")

    # Default format: (ref_id, ref_dur, ref_text, gen_id, gen_dur, gen_text)
    ref_id = fields[0]
    ref_text = fields[2].lower()
    gen_id = fields[3]
    gen_text = fields[5].lower()

    # Audio file path
    ref_audio_path = os.path.join(args.audio_root, ref_id + ".wav")

    # Output path
    out_path = os.path.join(args.output_dir, f"{gen_id}.wav")
    if os.path.exists(out_path):
        continue
    
    # TTS inference
    audio_segment, final_sr, _ = infer_process(
        ref_audio_path,
        ref_text,
        gen_text,
        model,
        vocoder,
        mel_spec_type=model_cfg.model.mel_spec.mel_spec_type,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
    )

    # Save output
    sf.write(out_path, audio_segment, final_sr)

    logger.info(f"{i} Saved: {out_path}")
