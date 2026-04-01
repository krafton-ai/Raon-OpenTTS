#!/usr/bin/env bash
# Download datasets for RAON-OpenTTS training.
#
# Usage:
#   bash scripts/data/download_datasets.sh --output_dir data/raw
#
# Requires: huggingface-cli (pip install huggingface-hub)

set -euo pipefail

OUTPUT_DIR="data/raw"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--output_dir DIR]"
            echo "  --output_dir  Directory to download datasets into (default: data/raw)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Downloading datasets to: ${OUTPUT_DIR}"
echo "============================================"

# 1. Emilia (~50K hours)
echo ""
echo "[1/13] Downloading Emilia ..."
huggingface-cli download amphion/Emilia \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/emilia" \
    --include "en_*" || echo "WARNING: Emilia download failed or incomplete"

# 2. Emilia-YODAS2 (~100K hours)
echo ""
echo "[2/13] Downloading Emilia-YODAS2 ..."
huggingface-cli download amphion/Emilia \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/emilia-yodas2" \
    --include "yodas2_en_*" || echo "WARNING: Emilia-YODAS2 download failed or incomplete"

# 3. LibriHeavy (~50K hours)
echo ""
echo "[3/13] Downloading LibriHeavy ..."
huggingface-cli download parler-tts/libriheavy \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/libriheavy" || echo "WARNING: LibriHeavy download failed or incomplete"

# 4. GigaSpeech (~10K hours)
echo ""
echo "[4/13] Downloading GigaSpeech ..."
huggingface-cli download speechcolab/gigaspeech \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/gigaspeech" || echo "WARNING: GigaSpeech download failed or incomplete"

# 5. VoxPopuli (~24K hours)
echo ""
echo "[5/13] Downloading VoxPopuli (English) ..."
huggingface-cli download facebook/voxpopuli \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/voxpopuli" \
    --include "en_*" || echo "WARNING: VoxPopuli download failed or incomplete"

# 6. People's Speech - clean (~12K hours)
echo ""
echo "[6/13] Downloading People's Speech (clean) ..."
huggingface-cli download MLCommons/peoples_speech \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/peoples_speech_clean" \
    --include "clean/*" || echo "WARNING: People's Speech (clean) download failed or incomplete"

# 7. People's Speech - dirty (~18K hours)
echo ""
echo "[7/13] Downloading People's Speech (dirty) ..."
huggingface-cli download MLCommons/peoples_speech \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/peoples_speech_dirty" \
    --include "dirty/*" || echo "WARNING: People's Speech (dirty) download failed or incomplete"

# 8. HiFi-TTS 2 (~10K hours)
echo ""
echo "[8/13] Downloading HiFi-TTS 2 ..."
huggingface-cli download reach-vb/hifi-tts-v2 \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/hifitts2" || echo "WARNING: HiFi-TTS 2 download failed or incomplete"

# 9. LibriTTS-R (~585 hours)
echo ""
echo "[9/13] Downloading LibriTTS-R ..."
huggingface-cli download cdminix/libritts-r-aligned \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/libritts-r" || echo "WARNING: LibriTTS-R download failed or incomplete"

# 10. SPGISpeech (~5K hours)
echo ""
echo "[10/13] Downloading SPGISpeech ..."
huggingface-cli download kensho/spgispeech \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/spgispeech" || echo "WARNING: SPGISpeech download failed or incomplete"

# 11. SPGISpeech2-Cut (~3K hours, derived from SPGISpeech)
echo ""
echo "[11/13] SPGISpeech2-Cut ..."
echo "  NOTE: This is an internally processed subset of SPGISpeech."
echo "  Run the SPGISpeech cutting pipeline after downloading SPGISpeech."

# 12. YouTube Commons (~200K hours)
echo ""
echo "[12/13] Downloading YouTube Commons (TTS-processed) ..."
huggingface-cli download KRAFTON/youtube-commons-tts \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/youtube" || echo "WARNING: YouTube Commons download failed or incomplete"

# 13. LibriLight text-corrected (~60K hours)
echo ""
echo "[13/13] Downloading LibriLight (text-corrected) ..."
huggingface-cli download KRAFTON/librilight-text-corrected \
    --repo-type dataset \
    --local-dir "${OUTPUT_DIR}/youtube-chunks" || echo "WARNING: LibriLight download failed or incomplete"

echo ""
echo "============================================"
echo "Download complete. Check ${OUTPUT_DIR} for results."
echo "============================================"
