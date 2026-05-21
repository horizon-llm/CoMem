set -x

project_name=summary-sft
experiment_name=summary-sft-qwen3-4b
save_path=checkpoints/${project_name}/${experiment_name}
mkdir -p $save_path

wandb login ${WANDB_API_KEY}
# Put the env's bin FIRST so it wins over base
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# Shift the arguments so $@ refers to the rest
shift 2

hf download YWZBrandon/summary_sft_data_v1 --repo-type dataset --local-dir outputs/summary_sft_data_v1

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=outputs/summary_sft_data_v1/data_with_summaries_train.parquet \
    data.val_files=outputs/summary_sft_data_v1/data_with_summaries_test.parquet \
    data.prompt_key=prompt \
    data.response_key=summary \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=32768 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen3-4B-Instruct-2507 \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=4 \
    use_remove_padding=True \
    ulysses_sequence_parallel_size=2 \
    model.strategy=fsdp \
    trainer.logger='["console","wandb"]' $@