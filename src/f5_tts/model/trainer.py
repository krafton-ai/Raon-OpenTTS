from __future__ import annotations

import gc
import logging
import math
import os
import torch

logger = logging.getLogger(__name__)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from .cfm import CFM
from .dataset import DynamicBatchSampler, collate_fn
from .utils import default, exists
import time 
from datetime import datetime
from torch import autocast
# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=0.5,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="raon-opentts",
        wandb_run_name="train",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan" | "sbhifigan16k"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        total_updates_per_epoch: int = None,  # total updates per epoch, null for auto from dataloader
        max_updates: int = None,  # stop training after this many updates (for fixed-step comparison)
        data_filtering: str = "all_data",  # all_data | dnsmos | wer | vad | combined
        reset_scheduler: bool = False,  # reset LR scheduler on resume (for extending training)
    ):
        self.total_updates_per_epoch = total_updates_per_epoch
        self.max_updates = max_updates
        self.data_filtering = data_filtering
        self.reset_scheduler = reset_scheduler
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        from accelerate.utils import InitProcessGroupKwargs
        from datetime import timedelta
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {
                    "wandb": {
                        "resume": "allow",
                        "name": wandb_run_name,
                        "id": wandb_resume_id,
                    }
                }
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            if self.accelerator.is_main_process:
                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")
            else:
                self.writer = None

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            logger.info("Using experiment tracker: %s", self.logger)
            if grad_accumulation_steps > 1:
                logger.info(
                    "Gradient accumulation uses per_updates counting (legacy checkpoints with per_steps are auto-converted)"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "checkpoints/default")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor


        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),  
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(
                    checkpoint, f"{self.checkpoint_path}/model_last.pt"
                )
                logger.info(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(
                    checkpoint, f"{self.checkpoint_path}/model_{update}.pt"
                )
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        logger.info(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(
                filename.endswith((".pt", ".safetensors"))
                for filename in os.listdir(self.checkpoint_path)
            )
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_"))
                and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [
                f
                for f in all_checkpoints
                if f.startswith("model_") and f != "model_last.pt"
            ]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(
                    f for f in all_checkpoints if f.startswith("pretrained_")
                )

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(
                f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu"
            )
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}",
                weights_only=True,
                map_location="cpu",
            )

        # Remove stale mel_spec keys from legacy checkpoints
        for key in [
            "ema_model.mel_spec.mel_stft.mel_scale.fb",
            "ema_model.mel_spec.mel_stft.spectrogram.window",
        ]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # Convert legacy per_steps checkpoints to per_updates
            if "step" in checkpoint:
                checkpoint["update"] = (
                    checkpoint["step"] // self.grad_accumulation_steps
                )
                if self.grad_accumulation_steps > 1 and self.is_main:
                    logger.warning(
                        "Loading legacy checkpoint with per_steps counting; auto-converting to per_updates."
                    )
            # Remove stale mel_spec keys from legacy checkpoints
            for key in [
                "mel_spec.mel_stft.mel_scale.fb",
                "mel_spec.mel_stft.spectrogram.window",
            ]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint["model_state_dict"]
            )


            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            if self.scheduler and not self.reset_scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            elif self.reset_scheduler:
                if self.is_main:
                    logger.info(f"[reset_scheduler] Skipping scheduler state load. Will fast-forward to step {checkpoint['update']}.")
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint["model_state_dict"]
            )
            update = 0

        del checkpoint
        gc.collect()
        return update
    
    def train(
        self, train_dataset: Dataset, length_list, num_workers=4, resumable_with_seed: int = None
    ):
        if self.log_samples:
            
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            from ..infer.utils_infer import (
                cfg_strength,
                load_vocoder,
                nfe_step,
                sway_sampling_coef,
            )

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.is_local_vocoder,
                local_path=self.local_vocoder_path,
            )
            target_sample_rate = self.accelerator.unwrap_model(
                self.model
            ).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        def build_train_dataloader_for_epoch(epoch: int):
            if self.batch_size_type == "sample":
                dl = DataLoader(
                    train_dataset,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=False,      # False since dataloader is rebuilt each epoch
                    batch_size=self.batch_size_per_gpu,
                    shuffle=True,
                    generator=generator,
                )
                return dl

            elif self.batch_size_type == "frame":
                self.accelerator.even_batches = False

                sampler = SequentialSampler(train_dataset)

                save_path = os.path.join(
                    self.checkpoint_path,
                    f"dynamic_batches_{self.data_filtering}_{self.batch_size_per_gpu}.json",
                )

                # max_padded_frames: cap on (max_frame_in_batch * num_samples)
                # prevents worst-case VRAM spikes from padding
                # default: frames_threshold * 3 (allows ~3x padding overhead)
                _max_padded = self.batch_size_per_gpu * 3

                # Only rank 0 generates batch file; others wait
                if not os.path.exists(save_path):
                    if self.accelerator.is_main_process:
                        logger.info(f"[Rank 0] Generating batch file: {save_path}")
                        DynamicBatchSampler(
                            sampler,
                            self.batch_size_per_gpu,
                            max_samples=self.max_samples,
                            random_seed=resumable_with_seed,
                            drop_residual=False,
                            save_path=save_path,
                            load_if_exists=False,
                            max_padded_frames=_max_padded,
                        )
                    self.accelerator.wait_for_everyone()

                batch_sampler = DynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    random_seed=resumable_with_seed,
                    drop_residual=False,
                    save_path=save_path,
                    max_padded_frames=_max_padded,
                )

                dl = DataLoader(
                    train_dataset,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    batch_sampler=batch_sampler,
                    #prefetch_factor=1,             # recommended 1 for stability if needed
                    multiprocessing_context="fork",
                )
                return dl

            else:
                raise ValueError(
                    f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}"
                )



        _probe_loader = build_train_dataloader_for_epoch(epoch=0)
        _probe_loader = self.accelerator.prepare_data_loader(_probe_loader)
        orig_epoch_step = len(_probe_loader)
        del _probe_loader
        gc.collect()


        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp


        if self.max_updates is not None:
            # Fix LR schedule to max_updates for fair comparison across experiments
            total_updates = self.max_updates * self.accelerator.num_processes
        elif self.total_updates_per_epoch is not None:
            total_updates = self.total_updates_per_epoch * self.epochs * self.accelerator.num_processes
        else:
            total_updates = orig_epoch_step * self.epochs

        logger.info("WARMUP UPDATES %d", warmup_updates)
        logger.info("TOTAL UPDATES %d", total_updates)
        decay_updates = total_updates - warmup_updates





        ## decaying after warmup

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        decay_scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates
        )
        hold_scheduler = ConstantLR(self.optimizer, factor=1e-8, total_iters=10**9)

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler, hold_scheduler],
            milestones=[warmup_updates, warmup_updates + decay_updates],
        )



        self.scheduler = self.accelerator.prepare(self.scheduler)


        start_update = self.load_checkpoint()
        global_update = start_update

        # Fast-forward scheduler to resume position when reset_scheduler is used
        if self.reset_scheduler and start_update > 0:
            if self.is_main:
                logger.info(f"[reset_scheduler] warmup_updates={warmup_updates}, total_updates={total_updates}, decay_updates={decay_updates}")
                logger.info(f"[reset_scheduler] start_update={start_update}, num_processes={self.accelerator.num_processes}")
            ff_steps = start_update
            if self.is_main:
                logger.info(f"[reset_scheduler] Fast-forwarding scheduler by {ff_steps} steps...")
            for _ in range(ff_steps):
                self.scheduler.step()
            if self.is_main:
                logger.info(f"[reset_scheduler] LR at resume: {self.scheduler.get_last_lr()[0]:.2e}")
                logger.info(f"[reset_scheduler] actual_lr: {self.optimizer.param_groups[0]['lr']:.2e}")

        if exists(resumable_with_seed):
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
        else:
            skipped_epoch = 0
            skipped_batch = 0


        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()


            train_dataloader = build_train_dataloader_for_epoch(epoch=epoch)
            train_dataloader = self.accelerator.prepare_data_loader(train_dataloader)
            if exists(resumable_with_seed) and epoch == skipped_epoch and skipped_batch > 0:
                current_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, num_batches=skipped_batch
                )
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
            else:
                current_dataloader = train_dataloader
                progress_bar_initial = 0


            # # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(
                train_dataloader.batch_sampler, "set_epoch"
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            time_per_accumstep = time.time()
            for batch in current_dataloader:

                if batch is None or (isinstance(batch, dict) and len(batch) == 0):
                    continue
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_start = time.time()
                    audio = (batch["audio"] if isinstance(batch["audio"], torch.Tensor) else torch.from_numpy(batch["audio"])).to(self.accelerator.device)
                    mel_spec = self.accelerator.unwrap_model(self.model).mel_spec(audio).permute(0, 2, 1)
                    if not torch.isfinite(mel_spec).all():
                        bad = batch.get("audio_paths", None)
                        logger.error("Non-finite mel_spec detected: %s", bad)
                        raise RuntimeError("mel_spec has NaN/Inf")

                    if not torch.isfinite(audio).all():
                        bad = batch.get("audio_paths", None)
                        logger.error("Non-finite audio detected: %s", bad)
                        raise RuntimeError("audio has NaN/Inf")

                    mel_lengths = (batch["mel_lengths"] if isinstance(batch["mel_lengths"], torch.Tensor) else torch.from_numpy(batch["mel_lengths"])).to(self.accelerator.device)#batch["mel_lengths"]

                    # NOTE: duration predictor training is reserved for future use
                    if (
                        self.duration_predictor is not None
                        and self.accelerator.is_local_main_process
                    ):
                        dur_loss = self.duration_predictor(
                            mel_spec, lens=batch.get("durations")
                        )
                        self.accelerator.log(
                            {"duration loss": dur_loss.item()}, step=global_update
                        )
                  
                    loss, cond, pred = self.model(
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
                        noise_scheduler=self.noise_scheduler,
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()

                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update), loss=loss.item()
                    )

                if self.accelerator.is_local_main_process:
                    actual_lr = self.optimizer.param_groups[0]["lr"]
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0], "actual_lr": actual_lr},
                        step=global_update,
                    )
                    if self.logger == "tensorboard" and self.writer is not None:
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar(
                            "lr", self.scheduler.get_last_lr()[0], global_update
                        )
                if (
                    self.last_per_updates < self.save_per_updates
                    and global_update % self.last_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update, last=True)

                # max_updates: stop training at fixed step count
                if self.max_updates is not None and global_update >= self.max_updates:
                    if self.is_main:
                        logger.info(f"Reached max_updates={self.max_updates}, stopping training.")
                    self.save_checkpoint(global_update, last=True)
                    self.accelerator.end_training()
                    return

                if (
                    global_update % self.save_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0]
                            + ([" "] if isinstance(text_inputs[0], list) else " ")
                            + text_inputs[0]
                        ]


                        with torch.inference_mode():
                            with autocast("cuda", dtype=torch.float16):
                                generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                    text=infer_text,
                                    duration=ref_audio_len * 2,
                                    steps=nfe_step,
                                    cfg_strength=cfg_strength,
                                    sway_sampling_coef=sway_sampling_coef,
                                )

                            generated = generated.to(torch.float32)
                            gen_mel_spec = (
                                generated[:, ref_audio_len:, :]
                                .permute(0, 2, 1)
                                .to(self.accelerator.device)
                            )
                            audio = (batch["audio"] if isinstance(batch["audio"], torch.Tensor) else torch.from_numpy(batch["audio"])).to(self.accelerator.device)#batch["audio"]
                            ref_mel_spec = self.accelerator.unwrap_model(self.model).mel_spec(audio)[0].unsqueeze(0)

                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name in ["bigvgan", "sbhifigan16k"]:
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        import soundfile as sf  # lazy import: only needed when log_samples is enabled
                        sf.write(
                            f"{log_samples_path}/update_{global_update}_gen.wav",
                            gen_audio.squeeze().numpy(),
                            target_sample_rate,
                        )
                        sf.write(
                            f"{log_samples_path}/update_{global_update}_ref.wav",
                            ref_audio.squeeze().numpy(),
                            target_sample_rate,
                        )
                        self.model.train()

            del current_dataloader
            del train_dataloader
            gc.collect()
        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
