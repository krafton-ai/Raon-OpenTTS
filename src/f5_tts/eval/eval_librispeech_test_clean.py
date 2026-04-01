# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import argparse
import json
import logging
import os

import multiprocessing as mp
from importlib.resources import files

import numpy as np

from .utils_eval import get_librispeech_test, run_asr_wer, run_sim

logger = logging.getLogger(__name__)


#rel_path = str(files("f5_tts").joinpath("../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="sim", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, default="output/librispeech_infer")
    parser.add_argument("-p", "--librispeech_test_clean_path", type=str, default="data/librispeech/test-clean")
    parser.add_argument("-n", "--gpu_nums", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    librispeech_test_clean_path = args.librispeech_test_clean_path  # test-clean path
    gen_wav_dir = args.gen_wav_dir
    metalst = "data/librispeech_pc_test_clean_cross_sentence.lst"

    gpus = list(range(args.gpu_nums))
    test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

    ## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
    ## leading to a low similarity for the ground truth in some cases.
    # test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

    asr_ckpt_dir = ""  # auto download to cache dir
    wavlm_ckpt_dir = "wavlm_large_finetune.pth"

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if eval_task == "wer":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                full_results.extend(r)
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
    with open(result_path, "w") as f:
        for line in full_results:
            metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(np.mean(metrics), 5)
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    logger.info(f"Total {len(metrics)} samples")
    logger.info(f"{eval_task.upper()}: {metric}")
    logger.info(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mp.set_start_method("spawn")
    main()