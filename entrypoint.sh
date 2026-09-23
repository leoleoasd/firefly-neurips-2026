# Example cluster orchestration script — adjust bucket names and paths to your environment.
DATA_BUCKET="${DATA_BUCKET:-your-s3-bucket}"
MODEL_BUCKET="${MODEL_BUCKET:-your-s3-bucket}"

bash launch_ray.sh

s5cmd cp -s --sp "s3://${DATA_BUCKET}/mcp-data/data_batches/\*" /data/mcp-data/data_batches/
s5cmd cp -s --sp "s3://${DATA_BUCKET}/mcp-data/mcp_servers_joined.json" /data/mcp-data/mcp_servers_joined.json
s5cmd cp -s --sp "s3://${MODEL_BUCKET}/base_models/Qwen/Qwen3-235B-A22B-Thinking-2507/\*" /tmp/instance_storage/Qwen/Qwen3-235B-A22B-Thinking-2507/


bash scripts/download_convert_model.sh \
     --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
     --config qwen3-30B-A3B

python tool_call_agent/convert_mcp_to_training.py \
        --tasks /data/mcp-data/data_batches/3/train_tasks_filtered.json \
        --servers /data/mcp-data/mcp_servers_joined.json \
        --output /data/mcp-data/data_batches/3/training_data.jsonl \
        --tool-mode dag

# if on main node:
python scripts/rm_router.py

ray job submit --address=auto \
  -- python scripts/sglang_job.py --num-gpus 1 --num-nodes 16 \
    --model /data/base_models/Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --context-length 131072  --reasoning-parser deepseek-r1  --tool-call-parser qwen



ray job submit --address=auto \
  --working-dir . \
  -- python scripts/sglang_job.py --num-gpus 4 --num-nodes 8 \
    --model /tmp/instance_storage/qwen3-30B-A3B \
    --context-length 131072  --reasoning-parser deepseek-r1  --tool-call-parser qwen
