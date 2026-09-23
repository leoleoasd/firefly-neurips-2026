import copy
import logging
import os
from pathlib import Path

import torch
from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

ORIGIN_SAMPLE_KEY = "origin_sample_key"


class RolloutDataSourceWithExclusion:
    """
    A self-contained data source that reads from a Dataset, supports a replay
    buffer, and can permanently exclude samples by their index in
    Dataset.origin_samples.

    Every sample returned by get_samples() carries
    metadata["origin_sample_key"] set to its index in Dataset.origin_samples.
    Call exclude_samples() with those indices to prevent the sample from being
    selected again in future rounds.
    """

    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        self.excluded_keys: set[int] = set()
        self.buffer: list[list[Sample]] = []

        if self.args.buffer_filter_path is None:
            self.buffer_filter = _pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            # Some text-only models ship a VLM-style HF config, so AutoProcessor
            # returns a real (multimodal) processor that breaks the text-only
            # prompt path. Set SLIME_DISABLE_PROCESSOR=1 to skip processor loading.
            if int(os.environ.get("SLIME_DISABLE_PROCESSOR", "0")):
                processor = None
            else:
                processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )

            # Tag each origin sample with its index. origin_samples is created
            # once and never replaced, so the tag is stable across shuffles.
            for i, sample in enumerate(self.dataset.origin_samples):
                sample.metadata[ORIGIN_SAMPLE_KEY] = i

            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    # ---- DataSource interface ------------------------------------------------

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        # 1. Drain buffer first
        buffer_samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(buffer_samples)
        if num_samples == 0:
            return buffer_samples

        # 2. Pull from dataset, skipping excluded
        prompt_samples: list[Sample] = []

        if self.dataset is not None:
            total = len(self.dataset.origin_samples)
            skipped = 0

            while len(prompt_samples) < num_samples:
                if self.sample_offset >= len(self.dataset):
                    self.epoch_id += 1
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(self.epoch_id)
                    self.sample_offset = 0

                candidate = self.dataset.samples[self.sample_offset]
                self.sample_offset += 1

                origin_key = candidate.metadata[ORIGIN_SAMPLE_KEY]
                if origin_key in self.excluded_keys:
                    skipped += 1
                    if skipped > total:
                        logger.warning("All samples have been excluded, no more data available")
                        break
                    continue

                skipped = 0
                prompt_samples.append(candidate)
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        # 3. Expand each prompt into a group of n_samples_per_prompt
        result = list(buffer_samples)
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            result.append(group)

        return result

    def add_samples(self, samples: list[list[Sample]]):
        if not samples:
            return
        for group in samples:
            assert isinstance(group, list)
            self.buffer.append(group)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "excluded_keys": list(self.excluded_keys),
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"Loading state from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        self.excluded_keys = set(state_dict.get("excluded_keys", []))

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

        logger.info(f"Restored {len(self.excluded_keys)} excluded samples")

    # ---- Exclusion API -------------------------------------------------------

    def exclude_samples(self, keys: list[int]):
        """Permanently exclude samples by their index in Dataset.origin_samples."""
        before = len(self.excluded_keys)
        self.excluded_keys.update(keys)
        after = len(self.excluded_keys)
        if after > before:
            logger.info(
                f"Excluded {after - before} new samples (total excluded: {after}/{len(self.dataset.origin_samples)})"
            )

    # ---- Buffer helpers ------------------------------------------------------

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []
        return self.buffer_filter(self.args, None, self.buffer, num_samples)

    def get_buffer_length(self):
        return len(self.buffer)


def _pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
