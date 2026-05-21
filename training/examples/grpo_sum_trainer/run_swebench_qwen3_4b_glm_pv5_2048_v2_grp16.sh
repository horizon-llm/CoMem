# this is the ablation to test the compressor prompt version v2

set -x

ENGINE=${1:-vllm}
INPUT_FILE="r2egym-training-trajectories-glm-t100"
ADDRESS=${ADDRESS:-"10.0.100.110"}
PORTS=${PORTS:-"8000"}
echo "Using port: $PORTS"
export VLLM_ATTENTION_BACKEND=XFORMERS

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please 'conda activate verl-agent' before running this script." >&2
  exit 1
fi

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

wandb login ${WANDB_API_KEY}

train_data_size=16
val_data_size=128
group_size=16
RUN_NAME="grpo_sum_qwen3_4b_glm_pv5_2048_sft_v2_grp16_max40"
EXP_NAME="verl_agent_swebench_sum_reward_v2"

#### preprocess data #####
bash scripts/preprocess_swebench_rollouts_to_parquet.sh $INPUT_FILE
#### preprocess data #####

mkdir -p checkpoints/${EXP_NAME}

LOCAL_INITIAL_MODEL=${LOCAL_INITIAL_MODEL:-"YWZBrandon/summary-sft-qwen3-4b"}

python3 -m verl.trainer.main_ppo \
    --config-path config \
    --config-name ppo_multi_agent_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=outputs/rollouts/${INPUT_FILE}_train_max40.parquet \
    data.val_files=outputs/rollouts/${INPUT_FILE}_test_max40.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=121072 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.prompt_key='trajectory_steps' \
    actor_rollout_ref.model.path=$LOCAL_INITIAL_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    env.swebench.use_tool_call=True \
    env.swebench.scaffold="r2egym" \
    env.rollout.n=$group_size \
    env.rollout.compressor_prompt_version="v5" \
    env.rollout.retain_most_turns=0 \
    env.traj_collection_type="sumapiresprecentlitellmglm" \
    env.response_agent.address=["http://$ADDRESS:$PORTS/v1"] \
    env.response_agent.model_name="hosted_vllm/zai-org/GLM-4.7" \
    env.response_agent.max_concurrency=128 \
    env.response_agent.max_executor_threads=128 \
    env.response_agent.per_endpoint_concurrency=64 \
    env.response_agent.timeout_s=80 \
    env.response_agent.temperature=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=5 \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True $@ | tee checkpoints/${EXP_NAME}/${RUN_NAME}.log
