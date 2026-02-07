#!/bin/bash

set -xeuo pipefail

export WORKING_DIR="${PWD}"

# Model
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
model_name="Qwen2.5-7B-Instruct"

# SAGE method
method="sage" # or "sage-light" for faster but slightly worse performance
hint_accuracy_min_threshold=0.0  # Only work for sage-light: When the accuracy of a prompt from previous epoch is lower than this threshold (hard), the hint level increases by 1.
hint_accuracy_max_threshold=0.35  # Only work for sage-light When the accuracy of a prompt from previous epoch is larger than this threshold (easy), the hint level decreases by 1.

# Wandb setting
project_name="SAGE"
experiment_name="${method}_${model_name}"
export WANDB_API_KEY=""
export WANDB_ENTITY=""
# export WANDB_MODE="offline"

# Output
ckpts_dir="./outputs/${experiment_name}"
mkdir -p "${ckpts_dir}/logs"
export WANDB_DIR=${ckpts_dir}/logs

# Trainig setting
NGPUS=8
train_prompt_bsz=128
train_prompt_mini_bsz=64

# Algorithm setting
algorithm=grpo
n=8
kl_coef=0.000
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Training data
train_path="./data/train.parquet"
test_path="./data/test.parquet"
train_files="['$train_path']"
test_files="['$test_path']"

python3 -m recipe.hint.main_hint \
    algorithm.adv_estimator=${algorithm} \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key="prompt" \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=${model_name_or_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.method=${method} \
    trainer.hint_accuracy_min_threshold=${hint_accuracy_min_threshold} \
    trainer.hint_accuracy_max_threshold=${hint_accuracy_max_threshold} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.test_freq=-1 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=1000 \
    trainer.resume_mode="auto" 2>&1 | tee ${ckpts_dir}/logs/log