#!/bin/bash


# Required ENVs (DATASET_PATH / CHECKPOINT_PATH / S3_BUCKET must point at your own S3 buckets):
# export JOB_NAME=qwen3b
# export DATASET_PATH=s3://your-bucket/rl_web_agent/sft_data_curation/data_fix/
# export JOB_ARGS="global_batch_size=64 micro_batch_size=2 chat_template=chat_templates/qwen25.jinja seq_length=1024 tensor_model_parallel_size=2"
# export CHECKPOINT_PATH=s3://your-bucket/rl_web_agent/sft_data_curation/checkpoints/$JOB_NAME
# export PROJECT_NAME=rl_web_agent_sft_curation
# export HF_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
# export MODEL_NAME=qwen25_3b

export TZ=America/Los_Angeles
export JOB_NAME=${JOB_NAME:-dummy}_$(date +%Y-%m-%d-%H-%M-%S)
export S3_BUCKET="${S3_BUCKET:?set S3_BUCKET}"
# today's date
export TODAY=$(date +%Y-%m-%d)
# project name, default to finetuned_web_agent only if it's undefined
export PROJECT_NAME=${PROJECT_NAME:-dummy}
export OUTPUT_DIR=${PROJECT_NAME}/logs/${TODAY}/${JOB_NAME}_${AWS_BATCH_JOB_NODE_INDEX}
export TMP_DIR=/tmp/instance_storage
export NEMO_HOME=$TMP_DIR/nemo
aws configure set region ap-south-1


main () {
    echo Will Output to $OUTPUT_DIR/output.log
    source .venv/bin/activate
    AWS_REGION=us-east-1  s5cmd cp --sp "${DATASET_PATH%/}/*" /workdir/sft_data/

    if [ $AWS_BATCH_JOB_NODE_INDEX -eq 0 ]; then
        export AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$(hostname -i)
    else
        export AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS
    fi
    echo "MASTER_ADDR: $AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"
    # yes | nemo llm import model=qwen25_7b source=hf://Qwen/Qwen2.5-7B-Instruct
    nemo llm import model=$MODEL_NAME source=hf://${HF_MODEL_NAME} --verbose -y
    # nemo nemo_finetuning train model=qwen25_3b chat_template=chat_templates/qwen25.jinja hf_model_name=Qwen/Qwen2.5-3B-Instruct seq_length=1024 tensor_model_parallel_size=2
    torchrun --nnodes $AWS_BATCH_JOB_NUM_NODES --nproc-per-node 8 --node-rank $AWS_BATCH_JOB_NODE_INDEX --master-addr $AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS --master-port 29500 --no-python \
            nemo nemo_finetuning train model=$MODEL_NAME hf_model_name=$HF_MODEL_NAME $JOB_ARGS
            # main.py $JOB_ARGS
    echo "train finished"
    AWS_REGION=us-east-1 s5cmd cp --sp "/tmp/instance_storage/checkpoints/*" ${CHECKPOINT_PATH%/}/raw_ckpt/
    torchrun --nnodes $AWS_BATCH_JOB_NUM_NODES --nproc-per-node 1 --node-rank $AWS_BATCH_JOB_NODE_INDEX --master-addr $AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS --master-port 29501 --no-python \
        echo "Checkpoint saved to ${CHECKPOINT_PATH%/}/raw_ckpt/"
    echo "checkpoint saved"
    if [ $AWS_BATCH_JOB_NODE_INDEX -eq 0 ]; then
        echo "main node merging checkpoints"
        AWS_REGION=us-east-1  s5cmd cp --sp "${CHECKPOINT_PATH%/}/raw_ckpt/*" /tmp/instance_storage/checkpoints_before_merge/
        echo "main node downloaded checkpoints"
        cd /tmp/instance_storage/checkpoints_before_merge/
        for i in */; do
            echo "Merging $i"
            nemo llm export path=$(pwd)/$i target=hf output_path=/tmp/instance_storage/checkpoints_after_merge/$i --verbose -y
            echo "Merged $i"
        done
        echo "main node merged checkpoints"
        AWS_REGION=us-east-1  s5cmd cp --sp /tmp/instance_storage/checkpoints_after_merge/ ${CHECKPOINT_PATH%/}/merged_ckpt/
    fi
}

main 2>&1 | tee-s3 --bucket $S3_BUCKET --key $OUTPUT_DIR/output.log --region ap-south-1
