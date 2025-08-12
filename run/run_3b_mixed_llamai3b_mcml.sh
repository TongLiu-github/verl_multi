#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:4
#SBATCH --output=06.08_llamai3b_mixed_v3_2epochs_12.08_lrz2.txt
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00


# model list: Qwen2.5-1.5B-Instruct, Qwen2.5-Math-7B,
# ---cpus-per-task=8
# -p lrz-hgx-h100-94x4, lrz-hgx-a100-80x4, lrz-dgx-a100-80x8
# -p mcml-hgx-h100-94x4, mcml-hgx-a100-80x4, mcml-dgx-a100-40x8
# -q mcml # -q mcml
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/tong/grpo/verl_multi

source ~/.bashrc
conda activate verl

echo "start to run"


## Qwen2.5-1.5/3B + GRPO + 75K
YOUR_PROJECT_NAME=r1-verl-grpo
GPUS_PER_NODE=4
#MODEL_PATH=Qwen/Qwen2.5-3B
#YOUR_RUN_NAME=r1training_grpo_qwen3B_mixed
MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
YOUR_RUN_NAME=r1training_grpo_llamai3b_mixed_v3_2epochs3
export HYDRA_FULL_ERROR=1

kl_coef=0.0
kl_loss_coef=0.0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/dss/dssfs05/pn39qo/pn39qo-dss-0001/tong/safety/verl/data/MATH-mixedprompt-v3/train.parquet \
    data.val_files=/dss/dssfs05/pn39qo/pn39qo-dss-0001/tong/safety/verl/data/MATH-500/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    +actor_rollout_ref.rollout.use_multi_prompt=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=290 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    trainer.save_rollout_generations=True \
    trainer.rollout_save_dir=/dss/dssfs05/pn39qo/pn39qo-dss-0001/tong/grpo/verl_multi/saved_rollout/$YOUR_RUN_NAME
