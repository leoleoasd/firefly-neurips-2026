import logging
import os

import click
import lightning as pl
import torch
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, NsysCallback
from nemo.utils.exp_manager import TimingCallback

logger = logging.getLogger(__name__)


@click.command()
@click.option("--seq_length", type=int, default=1024 * 128)
@click.option("--global_batch_size", type=int, default=1)
@click.option("--micro_batch_size", type=int, default=1)
@click.option("--num_steps", type=int, required=True)
@click.option("--max_epochs", type=int, required=True)
def main(seq_length, global_batch_size, micro_batch_size, num_steps, max_epochs):
    logging.basicConfig(level=logging.INFO, format=f"[rank={os.environ.get('RANK')}] %(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.DEBUG)
    qwen_25_chat_template = r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- generation -%}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
        {%- endgeneration -%}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    num_nodes = world_size // local_world_size
    print(f"{world_size=}, {local_world_size=}, {num_nodes=}")

    tokenizer = get_nmt_tokenizer(
        library="huggingface",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_fast=True,
        chat_template=qwen_25_chat_template,
    )
    model = llm.Qwen2Model(config=llm.Qwen25Config7B(), tokenizer=tokenizer)
    mcfg = model.config
    # mcfg.recompute_granularity = 'full'
    # mcfg.recompute_method = 'uniform'
    # mcfg.recompute_num_layers = 4  # as computed above
    mcfg.cpu_offload = True
    mcfg.cpu_offloading_num_layers = 31

    model.config.seq_length = seq_length

    data = llm.ChatDataModule(
        dataset_root="/tmp/instance_storage/sft_data",
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        tokenizer=tokenizer,
        dataset_kwargs={"pad_to_max_length": True, "get_attention_mask_from_fusion": True},
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        use_hf_tokenizer_chat_template=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="reduced_train_loss",
        save_last=False,
        every_n_train_steps=num_steps,
        dirpath="/tmp/instance_storage/checkpoints",
        always_save_context=True,
        save_top_k=-1,
    )

    strategy = nl.MegatronStrategy(
        context_parallel_size=world_size // 4,
        tensor_model_parallel_size=4,
        virtual_pipeline_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,
        ckpt_async_save=True,
    )

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-6,
        bf16=True,
    )

    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        num_nodes=num_nodes,
        devices=local_world_size,  ## you can change the number of devices to suit your setup
        max_steps=num_steps * max_epochs,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=0.0,
        callbacks=[
            NsysCallback(
                start_step=10,
                end_step=30,
            ),
            TimingCallback(),
        ],
    )
    wandb_logger = pl.pytorch.loggers.WandbLogger(project="rl_web_agent_sft_bootstrap")
    nemo_logger = nl.NeMoLogger(
        log_dir="/tmp/instance_storage/logdir",  ## logs and checkpoints will be written here
        wandb=wandb_logger,
        ckpt=checkpoint_callback,
    )
    wandb_logger.log_hyperparams(
        {
            "seq_length": seq_length,
            "global_batch_size": global_batch_size,
            "cpu_offload": model.config.cpu_offload if hasattr(model.config, "cpu_offload") else None,
            "cpu_offloading_num_layers": model.config.cpu_offloading_num_layers if hasattr(model.config, "cpu_offloading_num_layers") else None,
            "recompute_granularity": model.config.recompute_granularity,
            "recompute_method": model.config.recompute_method,
            "recompute_num_layers": model.config.recompute_num_layers,
        }
    )

    resume = nl.AutoResume(
        restore_config=nl.RestoreConfig(
            # Option A: use the imported HF model
            path="nemo://Qwen/Qwen2.5-7B-Instruct"
            # Option B: or a local path produced by llm.import_ckpt, e.g. "/models/Meta-Llama-3-8B"
        ),
        resume_if_exists=False,  # do not try to resume a prior training run
        resume_ignore_no_checkpoint=True,  # ok if there is no training ckpt yet
    )

    llm.train(
        # model=ckpt_path,
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    main()
