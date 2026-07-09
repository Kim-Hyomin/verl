#!/usr/bin/env bash
# Smoke test: verl official GRPO example (Qwen3-8B, FSDP, gsm8k only)
# Purpose: verify the pipeline runs on RTX PRO 6000 Blackwell (sm_120) before
#          swapping in AgenTracer data/reward.
set -euo pipefail

REPO=/home/intern/hyomin/verl
source "$REPO/.venv-verl/bin/activate"
cd "$REPO"

export HF_HOME=/home/intern/hyomin/.cache/huggingface
DATA="$REPO/data"
MODEL=/home/intern/hyomin/models/Qwen3-8B   # kept outside repo: 16GB

MACHINE=gb200 \
MODEL_PATH="$MODEL" \
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=16 \
ROLLOUT_N=4 \
ROLLOUT_TP=1 \
ROLLOUT_GPU_MEM_UTIL=0.5 \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=512 \
PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
PROJECT_NAME=verl_smoke \
EXPERIMENT_NAME=qwen3_8b_smoke \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files="['$DATA/gsm8k/train.parquet']" \
  data.val_files="['$DATA/gsm8k/test.parquet']" \
  trainer.logger='["console"]' \
  trainer.val_before_train=False \
  trainer.total_training_steps=2
