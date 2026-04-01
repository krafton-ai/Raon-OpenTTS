#!/bin/bash
# Usage: bash scripts/run_infer_eval.sh <checkpoint_path> [num_gpus] [config] [node]
# Example:
#   bash scripts/run_infer_eval.sh checkpoints/0.3B-emilia/model_3000.pt
#   bash scripts/run_infer_eval.sh checkpoints/0.3B-emilia/model_3000.pt 8 src/configs/03b_emilia.yaml my-node-01
#
# For multi-node Ray (e.g. 6 nodes x 8 GPUs = 48 GPUs):
#   bash scripts/run_infer_eval.sh checkpoints/model.pt 48 src/configs/03b.yaml my-node-01 6

set -euo pipefail

CKPT="${1:?Usage: $0 <checkpoint_path> [num_gpus] [config] [node] [num_nodes]}"
NUM_GPUS="${2:-8}"
CONFIG="${3:-src/configs/03b.yaml}"
HEAD_NODE="${4:-localhost}"
NUM_NODES="${5:-1}"

# Configure these paths for your environment
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV="${VENV:-${PROJ_DIR}/.venv/bin/python3}"
SCRIPT="${PROJ_DIR}/src/f5_tts/eval/run_infer_eval.py"

echo "=== TTS Infer + Eval ==="
echo "  CKPT:    $CKPT"
echo "  CONFIG:  $CONFIG"
echo "  GPUS:    $NUM_GPUS"
echo "  NODE:    $HEAD_NODE (+ $NUM_NODES node(s))"
echo "  PYTHON:  $VENV"
echo "========================"

# For single-node: just SSH and run
if [ "$NUM_NODES" -le 1 ]; then
    ssh -o StrictHostKeyChecking=no "$HEAD_NODE" bash -c "'
        export PYTHONPATH=${PROJ_DIR}/src:${PROJ_DIR}/src/f5_tts/eval
        cd ${PROJ_DIR}
        ${VENV} ${SCRIPT} \
            --ckpt ${CKPT} \
            --config ${CONFIG} \
            --num_gpus ${NUM_GPUS}
    '"
else
    # Multi-node Ray: start Ray head on HEAD_NODE, workers on other nodes
    HEAD_PORT=6379
    HEAD_NODE_ID="${HEAD_NODE##*-}"  # extract node number

    echo "Starting Ray head on $HEAD_NODE..."
    ssh -o StrictHostKeyChecking=no "$HEAD_NODE" bash -c "'
        export PYTHONPATH=${PROJ_DIR}/src:${PROJ_DIR}/src/f5_tts/eval
        ${PROJ_DIR}/.venv/bin/ray stop 2>/dev/null || true
        ${PROJ_DIR}/.venv/bin/ray start --head --port=${HEAD_PORT} --num-gpus=8
    '"
    HEAD_IP=$(ssh -o StrictHostKeyChecking=no "$HEAD_NODE" "hostname -I | awk '{print \$1}'")

    echo "Starting Ray workers on ${NUM_NODES} nodes..."
    for i in $(seq 2 "$NUM_NODES"); do
        worker_num=$((HEAD_NODE_ID + i - 1))
        worker_node="${HEAD_NODE%-*}-${worker_num}"
        echo "  Worker: $worker_node"
        ssh -o StrictHostKeyChecking=no "$worker_node" bash -c "'
            export PYTHONPATH=${PROJ_DIR}/src:${PROJ_DIR}/src/f5_tts/eval
            ${PROJ_DIR}/.venv/bin/ray stop 2>/dev/null || true
            ${PROJ_DIR}/.venv/bin/ray start --address=${HEAD_IP}:${HEAD_PORT} --num-gpus=8
        '" &
    done
    wait
    sleep 5

    echo "Running inference + eval on Ray cluster..."
    ssh -o StrictHostKeyChecking=no "$HEAD_NODE" bash -c "'
        export PYTHONPATH=${PROJ_DIR}/src:${PROJ_DIR}/src/f5_tts/eval
        export RAY_ADDRESS=${HEAD_IP}:${HEAD_PORT}
        cd ${PROJ_DIR}
        ${VENV} ${SCRIPT} \
            --ckpt ${CKPT} \
            --config ${CONFIG} \
            --num_gpus ${NUM_GPUS}
    '"

    echo "Stopping Ray..."
    for i in $(seq 1 "$NUM_NODES"); do
        node_num=$((HEAD_NODE_ID + i - 1))
        ssh -o StrictHostKeyChecking=no "${HEAD_NODE%-*}-${node_num}" "${PROJ_DIR}/.venv/bin/ray stop 2>/dev/null" &
    done
    wait
fi

echo "Done!"
