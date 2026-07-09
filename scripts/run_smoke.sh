#!/usr/bin/env bash
set -euo pipefail
REPO=/home/intern/hyomin/verl
source "$REPO/.venv-verl/bin/activate"
cd "$REPO"
export HF_HOME=/home/intern/hyomin/.cache/huggingface
DATA="$REPO/data"
export MACHINE=gb200
export MODEL_PATH=/home/intern/hyomin/models/Qwen3-8B
export NGPUS_PER_NODE=4
export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.5
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export PPO_MAX_TOKEN_LEN_PER_GPU=8192
export SAVE_FREQ=-1
export TEST_FREQ=-1
export PROJECT_NAME=verl_smoke
export EXPERIMENT_NAME=qwen3_8b_smoke
echo "DATA=$DATA"
ls -la "$DATA/gsm8k"
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files="['$DATA/gsm8k/train.parquet']" \
  data.val_files="['$DATA/gsm8k/test.parquet']" \
  trainer.logger='["console"]' \
  trainer.val_before_train=False \
  trainer.total_training_steps=2 \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa

