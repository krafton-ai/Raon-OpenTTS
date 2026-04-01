# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
try:
    from vocos import Vocos
except ImportError:
    Vocos = None

from ..model import CFM
from ..model.utils import convert_char_to_pinyin, get_tokenizer

_ref_audio_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------------------

target_sample_rate = 16000
n_mel_channels = 80
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if (
            len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8"))
            <= max_chars
        ):
            current_chunk += (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks




# load vocoder
def load_vocoder(
    vocoder_name="vocos",
    is_local=False,
    local_path="",
    device=device,
    hf_cache_dir=None,
):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            logger.info(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            logger.info("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(
                repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml"
            )
            model_path = hf_hub_download(
                repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin"
            )
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            logger.error(
                "You need to follow the README to init submodule and change the BigVGAN source code."
            )
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x",
                use_cuda_kernel=False,
                cache_dir=hf_cache_dir,
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "sbhifigan16k":
        from ..model.vocoder import load_hifigan_vocoder
        ckpt_path = os.path.join("pretrained_models", "tts-hifigan-libritts-16kHz", "generator.ckpt")
        if is_local and local_path:
            ckpt_path = os.path.join(local_path, "generator.ckpt")
        vocoder = load_hifigan_vocoder(ckpt_path, device=device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language}
        if language
        else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # Remove stale mel_spec keys from legacy checkpoints
        for key in [
            "mel_spec.mel_stft.mel_scale.fb",
            "mel_spec.mel_stft.spectrogram.window",
        ]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    vocab_char_map,
    vocab_size,
    mel_spec_type=mel_spec_type,
    ode_method=ode_method,
    use_ema=True,
    device=device,
):

    tokenizer = "custom"

    logger.info("token : %s", tokenizer)
    logger.info("model : %s", ckpt_path)

    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(
        audio, silence_threshold=silence_threshold
    )
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text
def fix_ref_text_ending(ref_text: str) -> str:
    ref_text = (ref_text or "").strip()
    if not ref_text:
        return ref_text
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    return ref_text

def estimate_ref_seconds_trimmed(
    ref_audio_path: str,
    base_silence_threshold: int = -42,
    target_dbfs: float = -20.0,
    max_gain_db: float = 30.0,
    thr_margin_db: float = 18.0,
):
    aseg = AudioSegment.from_file(ref_audio_path)

    # 1) Normalize for length estimation
    if aseg.dBFS != float("-inf"):
        gain = target_dbfs - aseg.dBFS
        gain = max(min(gain, max_gain_db), -max_gain_db)
        aseg_norm = aseg.apply_gain(gain)
    else:
        aseg_norm = aseg

    # 2) Dynamic threshold: treat audio below (mean dBFS - margin) as silence
    #    but clamp so it never exceeds the base threshold (-42)
    if aseg_norm.dBFS != float("-inf"):
        dyn_thr = aseg_norm.dBFS - thr_margin_db
        silence_threshold = min(base_silence_threshold, int(dyn_thr))
    else:
        silence_threshold = base_silence_threshold

    trimmed = remove_silence_edges(aseg_norm, silence_threshold=silence_threshold)
    trimmed = trimmed + AudioSegment.silent(duration=50)

    if len(trimmed) < 80:
        return aseg.duration_seconds
    return trimmed.duration_seconds

def preprocess_ref_audio_text(
    ref_audio_orig, ref_text, clip_short=True, show_info=print
):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=1000,
                seek_step=10,
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if (
                    len(non_silent_wave) > 6000
                    and len(non_silent_wave + non_silent_seg) > 12000
                ):
                    show_info("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=100,
                    silence_thresh=-40,
                    keep_silence=1000,
                    seek_step=10,
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if (
                        len(non_silent_wave) > 6000
                        and len(non_silent_wave + non_silent_seg) > 12000
                    ):
                        show_info("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        # if len(aseg) < 80:
        #     show_info(f"[SKIP] ref audio too short after trim: {len(aseg)}ms < {80}ms | {ref_audio_orig}")
        #     return None, None
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    logger.info("ref_text  %s", ref_text)

    return ref_audio, ref_text

import numpy as np

def normalize_peak(audio, eps=1e-3, max_gain=None):
    if isinstance(audio, torch.Tensor):
        max_abs = audio.abs().max()
        max_abs = torch.clamp(max_abs, min=eps)

        gain = 1.0 / max_abs
        if max_gain is not None:
            gain = torch.clamp(gain, max=max_gain)

        audio = audio * gain
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio

    # numpy path
    audio_np = np.asarray(audio)
    max_abs = np.max(np.abs(audio_np))
    max_abs = max(max_abs, eps)

    gain = 1.0 / max_abs
    if max_gain is not None:
        gain = min(gain, max_gain)

    audio_np = audio_np * gain
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return audio_np





def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    use_vad_duration: bool = True,
):
    """Run inference for a single (ref_audio, ref_text, gen_text) triple.

    Args:
        use_vad_duration: If True (default), use VAD-trimmed reference length
            only for generation-length estimation while conditioning on the
            original (non-trimmed) audio.  A dynamic silence threshold is
            applied so the estimate is robust to quiet speakers.
            If False, use the raw reference audio length for both conditioning
            and generation-length estimation (original F5-TTS behaviour).
    """
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    audio = normalize_peak(audio)

    ref_text = fix_ref_text_ending(ref_text)
    if use_vad_duration:
        ref_seconds_for_length = estimate_ref_seconds_trimmed(ref_audio)
    else:
        ref_seconds_for_length = None  # fall back to raw audio length

    max_chars = int(
        len(ref_text.encode("utf-8"))
        / max(ref_seconds_for_length, 1e-6)
        * (22 - ref_seconds_for_length)
    )
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)



    for i, gen_text in enumerate(gen_text_batches):
        logger.info("gen_text %d: %s", i, gen_text)

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            ref_seconds_for_length=ref_seconds_for_length,
            use_vad_duration=use_vad_duration,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    ref_seconds_for_length=None,
    use_vad_duration: bool = True,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    audio = audio.to(device)
    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)
        ref_audio_len_cond = audio.shape[-1] // hop_length

        # (2) Ref length for duration estimation: based on trimmed seconds
        if ref_seconds_for_length is not None:
            ref_audio_len_est = int(ref_seconds_for_length * target_sample_rate / hop_length)
        else:
            ref_audio_len_est = ref_audio_len_cond


        if fix_duration is not None:
            # Interpret fix_duration as "generation segment length in seconds"
            gen_len = int(fix_duration * target_sample_rate / hop_length)
        else:
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))


            # ref_audio_len_est (frames) -> seconds
            ref_sec_est = ref_audio_len_est * hop_length / target_sample_rate
            sec_per_byte = ref_sec_est / max(ref_text_len, 1)

            if use_vad_duration:
                # Clamp to guarantee a minimum speech rate (12 chars/sec).
                # Prevents excessively slow/long generation for quiet or short prompts.
                MIN_CHARS_PER_SEC = 12.0
                sec_per_byte = min(sec_per_byte, 1.0 / MIN_CHARS_PER_SEC)

            gen_sec = (sec_per_byte * gen_text_len) / max(local_speed, 1e-6)
            gen_len = int(gen_sec * target_sample_rate / hop_length)

            # gen_len = int(ref_audio_len_est / max(ref_text_len, 1) * gen_text_len / local_speed)

        # Prevent zero frames (generate at least 1 frame)
        gen_len = max(gen_len, 1)

        # Total duration = (original cond length) + (predicted generation segment length)
        duration = ref_audio_len_cond + gen_len
      
        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len_cond:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            else:
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in (
            progress.tqdm(gen_text_batches)
            if progress is not None
            else gen_text_batches
        ):
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_batch, gen_text)
                for gen_text in gen_text_batches
            ]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(
                        cross_fade_samples, len(prev_wave), len(next_wave)
                    )

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = (
                        prev_overlap * fade_out + next_overlap * fade_in
                    )

                    # Combine
                    new_wave = np.concatenate(
                        [
                            prev_wave[:-cross_fade_samples],
                            cross_faded_overlap,
                            next_wave[cross_fade_samples:],
                        ]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
