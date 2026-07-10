#!/usr/bin/env bash
# Probe run: verify rollout format + reward wiring. NOT a training run.
set -euo pipefail

REPO=/home/intern/hyomin/verl
source "$REPO/.venv-verl/bin/activate"
cd "$REPO"
unset CUDA_VISIBLE_DEVICES

export HF_HOME=/home/intern/hyomin/.cache/huggingface
DATA="$REPO/data"

export MACHINE=gb200
export MODEL_PATH=/home/intern/hyomin/models/Qwen3-8B
export NGPUS_PER_NODE=4
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export ROLLOUT_N=8
export ROLLOUT_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.3
export MAX_PROMPT_LENGTH=12288
export MAX_RESPONSE_LENGTH=8192
export PPO_MAX_TOKEN_LEN_PER_GPU=20480
export SAVE_FREQ=-1
export TEST_FREQ=-1
export PROJECT_NAME=agentracer_probe
export EXPERIMENT_NAME=train5

mkdir -p logs/rollouts

bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files="['$DATA/agentracer/train.parquet']" \
  data.val_files="['$DATA/agentracer/dev.parquet']" \
  data.dataloader_num_workers=2 \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  reward.custom_reward_function.path="$REPO/agentracer/reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.rollout_data_dir="$REPO/logs/rollouts" \
  trainer.logger='["console"]' \
  trainer.val_before_train=False \
  trainer.total_training_steps=5 \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.rollout.multi_stage_wake_up=True

