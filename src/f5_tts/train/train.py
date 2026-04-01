from importlib.resources import files
import logging
import os

import torch

logger = logging.getLogger(__name__)
import hydra
from omegaconf import OmegaConf

from ..model import CFM, Trainer
from ..model.dataset import load_raon_pool
from ..model.utils import get_tokenizer

os.chdir(
    str(files("f5_tts").joinpath("../.."))
)  # change working directory to root of project (local editable)


@hydra.main(
    version_base="1.3",
    config_path=str(files("f5_tts").joinpath("configs")),
    config_name=None,
)
def main(model_cfg):
    model_cls = hydra.utils.get_class(
        f"f5_tts.model.{model_cfg.model.backbone}"
    )

    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}"
    wandb_resume_id = None

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # tokenizer: "custom" → reads vocab from model.tokenizer_path directly
    # tokenizer: "byte"   → 256 vocab, no external file
    tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(
            files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")
        ),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="raon-opentts",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        total_updates_per_epoch=model_cfg.optim.total_updates_per_epoch,
        max_updates=OmegaConf.select(model_cfg, "optim.max_updates", default=None),
        data_filtering=model_cfg.datasets.split,
        reset_scheduler=OmegaConf.select(model_cfg, "optim.reset_scheduler", default=False),
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = model_cfg.datasets
    nonredist_dirs = OmegaConf.to_container(
        OmegaConf.select(model_cfg, "datasets.nonredist_dirs", default=[]) or [],
        resolve=True,
    )

    train_dataset, durations = load_raon_pool(
        hf_repo=ds_cfg.hf_repo,
        local_cache_dir=OmegaConf.select(model_cfg, "datasets.local_cache_dir", default=None),
        configs=OmegaConf.to_container(
            OmegaConf.select(model_cfg, "datasets.configs", default=None) or [],
            resolve=True,
        ) or None,
        split=ds_cfg.split,
        nonredist_dirs=nonredist_dirs,
        target_sample_rate=model_cfg.model.mel_spec.target_sample_rate,
        hop_length=model_cfg.model.mel_spec.hop_length,
    )

    trainer.train(
        train_dataset,
        durations,
        num_workers=ds_cfg.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
