export JOB_NAME=qwen3b
# Fill in your own bucket/paths before sourcing this file.
export DATASET_PATH=s3://your-bucket/rl_web_agent/sft_data_curation/data_fix/
export JOB_ARGS="global_batch_size=64 micro_batch_size=2 chat_template=chat_templates/qwen25.jinja tensor_model_parallel_size=2"
export CHECKPOINT_PATH=s3://your-bucket/rl_web_agent/sft_data_curation/checkpoints/$JOB_NAME
export PROJECT_NAME=rl_web_agent_sft_curation
export HF_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
export MODEL_NAME=qwen25_3b
