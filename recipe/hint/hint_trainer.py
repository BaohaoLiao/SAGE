# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import json5
import os
import uuid
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.torch_functional import postprocess_data
from verl.utils.model import compute_position_id_with_mask

from recipe.hint.reward_tracker import RewardTracker
from recipe.hint.prompt import ANSWER_SYSTEM_PROMPT, HINT_SYSTEM_PROMPT, HINT_USER_PROMPT_TEMPLATE


class RayHintTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Initialize reward tracker for monitoring data point rewards
        self.reward_tracker = RewardTracker()

        # Select hinting strategy: "sage" (on-policy) or "sage-light" (original/off-policy-like)
        self.hint_method = self.config.trainer.get("method", "sage-light")
        if self.hint_method not in {"sage", "sage-light"}:
            raise ValueError(f"Unsupported trainer.method '{self.hint_method}'. Choose 'sage' or 'sage-light'.")

    def _build_hint_messages(self, question: str, solution: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": HINT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HINT_USER_PROMPT_TEMPLATE.format(problem=question, solution=solution),
            },
        ]

    def _prepare_prompt_inputs(
        self, messages: list[dict[str, str]]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], str]]:
        try:
            apply_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            input_ids, attention_mask = postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.get("truncation", "error"),
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            raw_prompt_ids = [tid for tid, mask_val in zip(input_ids[0].tolist(), attention_mask[0].tolist()) if mask_val]
            return input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt
        except Exception as e:
            print(f"Failed to build prompt inputs: {e}")
            return None
        
    def _parse_hint_payload(self, hint_text: str) -> Optional[dict[str, Any]]:
        # Normalize escaped newlines to actual newlines so fence detection works.
        stripped = hint_text #.replace("\\n", "\n").strip()
        # Prefer content inside ```json ... ``` fences when present.
        json_fence = stripped.lower().find("```json")
        if json_fence != -1:
            end = stripped.find("```", json_fence + 6)
            if end != -1:
                stripped = stripped[json_fence + 6 : end].strip()
        elif stripped.startswith("```") and stripped.count("```") >= 2:
            start = stripped.find("```")
            end = stripped.find("```", start + 3)
            if end != -1:
                stripped = stripped[start + 3 : end].strip()
        # Heuristic: if there is a JSON object after some intro text, find first brace.
        brace_pos = stripped.find("{")
        if brace_pos > 0:
            stripped_candidate = stripped[brace_pos:]
        else:
            stripped_candidate = stripped
        try:
            return json5.loads(stripped_candidate)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json5.loads(stripped[start : end + 1])
                except Exception:
                    return None
            return None
        except Exception:
            return None
        
    def _generate_hints_batch(
        self, requests: list[tuple[int, str, str]], max_retries: int = 3
    ) -> tuple[dict[int, dict[str, Any]], int, int, dict[int, str]]:
        """Generate full hint payloads (all levels) for multiple questions in one batch with retries."""
        prepared: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]] = {}
        for idx, question, solution in requests:
            prompt_inputs = self._prepare_prompt_inputs(self._build_hint_messages(question, solution))
            if prompt_inputs is None:
                continue
            prepared[idx] = prompt_inputs[:4]

        if not prepared:
            return {}, 0, 0, {}

        hints: dict[int, dict[str, Any]] = {}
        raw_hints: dict[int, str] = {}
        remaining = list(prepared.keys())
        attempt = 0
        while remaining and attempt < max_retries:
            tensors = [prepared[idx] for idx in remaining]
            input_ids = torch.cat([item[0] for item in tensors], dim=0)
            attention_mask = torch.cat([item[1] for item in tensors], dim=0)
            position_ids = torch.cat([item[2] for item in tensors], dim=0)
            raw_prompt_ids = np.array([item[3] for item in tensors], dtype=object)

            hint_batch = DataProto.from_dict(
                tensors={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                non_tensors={"raw_prompt_ids": raw_prompt_ids},
                meta_info={"global_steps": self.global_steps},
            )
            hint_batch.meta_info["recompute_log_prob"] = False
            hint_batch.meta_info["validate"] = True
            hint_batch.meta_info["do_sample"] = True

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            hint_batch_padded, pad_size = pad_dataproto_to_divisor(hint_batch, size_divisor)
            if not self.async_rollout_mode:
                hint_output_padded = self.actor_rollout_wg.generate_sequences(hint_batch_padded)
            else:
                hint_output_padded = self.async_rollout_manager.generate_sequences(hint_batch_padded)
            hint_output = unpad_dataproto(hint_output_padded, pad_size=pad_size)

            if "responses" not in hint_output.batch.keys():
                break

            next_remaining = []
            for i, idx in enumerate(remaining):
                hint_response = hint_output.batch["responses"][i]
                hint_text = self.tokenizer.decode(hint_response, skip_special_tokens=True).strip()
                raw_hints[idx] = hint_text
                hint_payload = self._parse_hint_payload(hint_text)
                if hint_payload:
                    hints[idx] = hint_payload
                else:
                    preview = [hint_text]
                    print(f"[hint_parse_failed] idx={idx} attempt={attempt} text_preview={preview}")
                    next_remaining.append(idx)

            remaining = next_remaining
            attempt += 1
        failed = len(remaining)
        return hints, failed, attempt, raw_hints
    
    def _build_answer_messages_with_hint(self, question: str, hint_text: str) -> list[dict[str, str]]:
        user_content = (
            f"Question:\n{question}\n\nHere is a hint to help you:\n{hint_text}"
        )
        return [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _extract_question_and_solution(
        self, base_batch: DataProto, gen_batch: DataProto, idx: int
    ) -> tuple[Optional[str], Optional[str]]:
        non_tensor_batch = base_batch.non_tensor_batch

        question = None
        for key in ("problem", "question"):
            if key in non_tensor_batch:
                value = non_tensor_batch[key][idx]
                if isinstance(value, str) and value.strip():
                    question = value.strip()
                    break

        solution = None
        if "solution" in non_tensor_batch:
            value = non_tensor_batch["solution"][idx]
            if isinstance(value, str) and value.strip():
                solution = value.strip()
        if solution is None and "answer" in non_tensor_batch:
            value = non_tensor_batch["answer"][idx]
            if isinstance(value, str) and value.strip():
                solution = value.strip()
        if solution is None and "reward_model" in non_tensor_batch:
            reward_info = non_tensor_batch["reward_model"][idx]
            if isinstance(reward_info, dict):
                for key in ("solution", "ground_truth", "answer"):
                    value = reward_info.get(key)
                    if isinstance(value, str) and value.strip():
                        solution = value.strip()
                        break

        if question is None and "raw_prompt_ids" in gen_batch.non_tensor_batch:
            try:
                question = self.tokenizer.decode(gen_batch.non_tensor_batch["raw_prompt_ids"][idx], skip_special_tokens=True).strip()
            except Exception:
                question = None

        return question, solution
    
    def _apply_hints_to_gen_batch(
        self,
        gen_batch: DataProto,
        base_batch: DataProto,
        need_hint_indices: list[int],
        hint_payloads: dict[int, dict[str, Any]],
        level_key: str,
    ) -> Optional[tuple[DataProto, list[int]]]:
        if not need_hint_indices:
            return None

        requests: list[tuple[int, str, str]] = []
        question_map: dict[int, str] = {}
        for idx in need_hint_indices:
            question, solution = self._extract_question_and_solution(base_batch, gen_batch, idx)
            if question and solution:
                requests.append((idx, question, solution))
                question_map[idx] = question

        updated_gen_batch = deepcopy(gen_batch)
        applied_indices: list[int] = []

        for idx in need_hint_indices:
            payload = hint_payloads.get(idx, {})
            hint_text = payload.get(level_key, "")
            if not hint_text:
                continue
            question = question_map.get(idx)
            if not question:
                continue
            prompt_inputs = self._prepare_prompt_inputs(self._build_answer_messages_with_hint(question, hint_text))
            if prompt_inputs is None:
                continue
            input_ids, attention_mask, position_ids, raw_prompt_ids, _ = prompt_inputs

            updated_gen_batch.batch["input_ids"][idx] = input_ids[0]
            updated_gen_batch.batch["attention_mask"][idx] = attention_mask[0]
            updated_gen_batch.batch["position_ids"][idx] = position_ids[0]

            if "raw_prompt_ids" not in updated_gen_batch.non_tensor_batch:
                updated_gen_batch.non_tensor_batch["raw_prompt_ids"] = np.array(
                    [None for _ in range(len(updated_gen_batch))], dtype=object
                )
            updated_gen_batch.non_tensor_batch["raw_prompt_ids"][idx] = raw_prompt_ids
            applied_indices.append(idx)

        if not applied_indices:
            return None
        return updated_gen_batch, applied_indices
    
    def _clone_dataproto(self, data: DataProto) -> DataProto:
        tensors = data.batch.clone() if data.batch is not None else None
        non_tensors = {k: deepcopy(v) for k, v in data.non_tensor_batch.items()}
        meta_info = deepcopy(data.meta_info)
        if tensors is not None:
            return DataProto(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info)
        return DataProto.from_dict(tensors={}, non_tensors=non_tensors, meta_info=meta_info)

    def _replace_slices(self, target: DataProto, source: DataProto, indices: list[int], repeat: int):
        """Replace rollout slices for specific questions from source into target."""
        for idx in indices:
            start = idx * repeat
            end = start + repeat
            for key in target.batch.keys():
                target.batch[key][start:end] = source.batch[key][start:end]
            for key in target.non_tensor_batch.keys():
                if key in source.non_tensor_batch:
                    target.non_tensor_batch[key][start:end] = source.non_tensor_batch[key][start:end]

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        hint_context_keys = {"problem", "question", "solution", "answer", "index"}

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys - hint_context_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save reward tracker
        self.reward_tracker.save_checkpoint(local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load reward tracker
        self.reward_tracker.load_checkpoint(global_step_folder)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                base_batch = self._clone_dataproto(batch)
                base_gen_batch = self._clone_dataproto(gen_batch)
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    rollout_repeat = self.config.actor_rollout_ref.rollout.n

                    def run_rollout(gen_batch_for_run: DataProto):
                        gen_batch_output = gen_batch_for_run.repeat(repeat_times=rollout_repeat, interleave=True)

                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output_out = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output_out = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            timing_raw.update(gen_batch_output_out.meta_info["timing"])
                            gen_batch_output_out.meta_info.pop("timing", None)

                        working_batch = self._clone_dataproto(base_batch)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = self._clone_dataproto(gen_batch_for_run)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                        gen_baseline_batch
                                    )
                                working_batch = working_batch.union(gen_baseline_output)
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in working_batch.batch.keys():
                                    rm_scores = self.rm_wg.compute_rm_score(working_batch)
                                    working_batch = working_batch.union(rm_scores)
                                reward_baseline_tensor, _ = compute_reward(working_batch, self.reward_fn)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                working_batch.pop(batch_keys=list(keys_to_pop))

                                working_batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output

                        working_batch = working_batch.repeat(repeat_times=rollout_repeat, interleave=True)
                        working_batch = working_batch.union(gen_batch_output_out)

                        if "response_mask" not in working_batch.batch.keys():
                            working_batch.batch["response_mask"] = compute_response_mask(working_batch)
                        if self.config.trainer.balance_batch:
                            self._balance_batch(working_batch, metrics=metrics)

                        working_batch.meta_info["global_token_num"] = torch.sum(
                            working_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm and "rm_scores" not in working_batch.batch.keys():
                                reward_tensor_out = self.rm_wg.compute_rm_score(working_batch)
                                working_batch = working_batch.union(reward_tensor_out)
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor_out, reward_extra_infos_out = ray.get(
                                    compute_reward_async.remote(
                                        data=working_batch, config=self.config, tokenizer=self.tokenizer
                                    )
                                )
                            else:
                                reward_tensor_out, reward_extra_infos_out = compute_reward(
                                    working_batch, self.reward_fn
                                )

                        return working_batch, reward_tensor_out, reward_extra_infos_out

                    index_array = base_batch.non_tensor_batch.get("index")
                    min_threshold = self.config.trainer.get("hint_accuracy_min_threshold", 0)
                    max_threshold = self.config.trainer.get("hint_accuracy_max_threshold", 0.35)
                    if min_threshold > max_threshold:
                        min_threshold, max_threshold = max_threshold, min_threshold

                    if self.hint_method == "sage":
                        # On-policy hinting ("sage"): run once, hint unresolved prompts, reroll slices, merge metrics
                        batch, reward_tensor, reward_extra_infos_dict = run_rollout(base_gen_batch)

                        sequence_rewards = reward_tensor.sum(dim=-1) if reward_tensor is not None else None
                        rewards_per_question = (
                            sequence_rewards.reshape(len(base_batch), rollout_repeat)
                            if sequence_rewards is not None
                            else None
                        )
                        resolved_mask = (
                            torch.any(rewards_per_question > 0, dim=1) if rewards_per_question is not None else None
                        )

                        hint_payloads: dict[int, dict[str, Any]] = {}
                        hint_payloads_raw: dict[int, str] = {}
                        hint_failed = 0
                        hint_attempts = 0
                        hint_applied_prompts: set[int] = set()
                        hint_final_level: dict[int, str] = {}
                        effective_level_by_batch_idx = {idx: "no_hint" for idx in range(len(base_batch))}

                        if rewards_per_question is not None and not torch.all(resolved_mask):
                            metrics["hint/prompts_needing_hint"] = int((~resolved_mask).sum().item())
                            requests = []
                            for idx in range(len(base_batch)):
                                if not resolved_mask[idx]:
                                    question, solution = self._extract_question_and_solution(
                                        base_batch, base_gen_batch, idx
                                    )
                                    if question and solution:
                                        requests.append((idx, question, solution))

                            hint_payloads, hint_failed, hint_attempts, hint_payloads_raw = self._generate_hints_batch(
                                requests
                            )
                            metrics["hint/generate_failed"] = hint_failed
                            metrics["hint/generate_attempts"] = hint_attempts
                            metrics["hint/generate_requested"] = len(requests)
                            if hint_payloads:
                                metrics["hint/used"] = 1
                                metrics["hint/num_questions"] = len(hint_payloads)
                                metrics["hint/generate_success"] = len(hint_payloads)
                                if "index" in base_batch.non_tensor_batch:
                                    self.reward_tracker.log_hint_raw(
                                        base_batch.non_tensor_batch["index"],
                                        {idx: raw for idx, raw in hint_payloads_raw.items()},
                                        self.global_steps,
                                        used=False,
                                        failed=False,
                                    )
                                    self.reward_tracker.log_hint_payloads(
                                        base_batch.non_tensor_batch["index"],
                                        hint_payloads,
                                        self.global_steps,
                                        used=False,
                                        failed=False,
                                    )
                            else:
                                if "index" in base_batch.non_tensor_batch and hint_failed:
                                    failed_payloads = {idx: {} for idx in requests}
                                    self.reward_tracker.log_hint_payloads(
                                        base_batch.non_tensor_batch["index"],
                                        failed_payloads,
                                        self.global_steps,
                                        used=False,
                                        failed=True,
                                    )
                                    self.reward_tracker.log_hint_raw(
                                        base_batch.non_tensor_batch["index"],
                                        {idx: hint_payloads_raw.get(idx, "") for idx in requests},
                                        self.global_steps,
                                        used=False,
                                        failed=True,
                                    )

                            for level_key in ["level_1", "level_2", "level_3"]:
                                target_indices = [
                                    idx
                                    for idx in range(len(base_batch))
                                    if not resolved_mask[idx]
                                    and idx in hint_payloads
                                    and isinstance(hint_payloads[idx].get(level_key, ""), str)
                                    and hint_payloads[idx].get(level_key, "").strip()
                                ]
                                if not target_indices:
                                    continue

                                hinted = self._apply_hints_to_gen_batch(
                                    base_gen_batch, base_batch, target_indices, hint_payloads, level_key
                                )
                                if hinted is None:
                                    continue
                                gen_batch_to_use, applied_indices = hinted

                                candidate_batch, candidate_reward_tensor, _ = run_rollout(gen_batch_to_use)

                                self._replace_slices(batch, candidate_batch, applied_indices, rollout_repeat)
                                for idx in applied_indices:
                                    start = idx * rollout_repeat
                                    end = start + rollout_repeat
                                    reward_tensor[start:end] = candidate_reward_tensor[start:end]

                                if applied_indices:
                                    hint_applied_prompts.update(applied_indices)
                                    for idx in applied_indices:
                                        hint_final_level[idx] = level_key
                                        effective_level_by_batch_idx[idx] = level_key

                                if "index" in base_batch.non_tensor_batch:
                                    payload_subset = {
                                        idx: hint_payloads[idx] for idx in applied_indices if idx in hint_payloads
                                    }
                                    if payload_subset:
                                        raw_subset = {idx: hint_payloads_raw.get(idx, "") for idx in applied_indices}
                                        self.reward_tracker.log_hint_raw(
                                            base_batch.non_tensor_batch["index"],
                                            raw_subset,
                                            self.global_steps,
                                            used=True,
                                            failed=False,
                                        )
                                        self.reward_tracker.log_hint_payloads(
                                            base_batch.non_tensor_batch["index"],
                                            payload_subset,
                                            self.global_steps,
                                            used=True,
                                            failed=False,
                                        )

                                sequence_rewards = reward_tensor.sum(dim=-1)
                                rewards_per_question = sequence_rewards.reshape(len(base_batch), rollout_repeat)
                                resolved_mask = torch.any(rewards_per_question > 0, dim=1)

                            reward_extra_infos_dict = {}

                        metrics["hint/prompts_with_hint"] = len(hint_applied_prompts)
                        level_counts = {"level_1": 0, "level_2": 0, "level_3": 0}
                        for level in hint_final_level.values():
                            if level in level_counts:
                                level_counts[level] += 1
                        for level_key, count in level_counts.items():
                            metrics[f"hint/used_{level_key}"] = count
                        metrics["hint/generate_success"] = metrics.get("hint/generate_success", len(hint_final_level))

                        accuracies = []
                        if rewards_per_question is not None:
                            correct_mask = rewards_per_question > 0
                            accuracies = correct_mask.float().mean(dim=1).cpu().tolist()

                            level_accumulators = {"no_hint": [], "level_1": [], "level_2": [], "level_3": []}
                            for idx, acc in enumerate(accuracies):
                                level = effective_level_by_batch_idx.get(idx, "no_hint")
                                level_accumulators[level].append(acc)

                            for level_key, accs in level_accumulators.items():
                                metrics[f"hint/used_{level_key}"] = metrics.get(f"hint/used_{level_key}", 0) + len(accs)
                                if accs:
                                    metrics[f"hint/acc_{level_key}_mean"] = float(np.mean(accs))

                            if index_array is not None:
                                self.reward_tracker.log_hint_accuracy(
                                    index_array, effective_level_by_batch_idx, accuracies, self.global_steps
                                )
                                metrics["hint/accuracy_min_threshold"] = float(min_threshold)
                                metrics["hint/accuracy_max_threshold"] = float(max_threshold)
                                for idx, acc in enumerate(accuracies):
                                    index_str = str(index_array[idx])
                                    current_level = effective_level_by_batch_idx.get(idx, "no_hint")
                                    self.reward_tracker.set_hint_level(index_str, current_level)
                                    self.reward_tracker.set_last_hint_accuracy(index_str, acc)
                    else:
                        # Original (off-policy style) hint logic ("sage-light")
                        current_level_by_batch_idx: dict[int, str] = {}
                        level_to_indices = {"level_1": [], "level_2": [], "level_3": []}
                        for idx in range(len(base_batch)):
                            if index_array is not None:
                                index_str = str(index_array[idx])
                                prev_level = self.reward_tracker.get_hint_level(index_str, "no_hint")
                                prev_acc = self.reward_tracker.get_last_hint_accuracy(index_str)
                            else:
                                prev_level = "no_hint"
                                prev_acc = None

                            if prev_acc is None:
                                level = "no_hint"
                            else:
                                level = prev_level
                                if prev_acc <= min_threshold:
                                    if prev_level == "no_hint":
                                        level = "level_1"
                                    elif prev_level == "level_1":
                                        level = "level_2"
                                    elif prev_level == "level_2":
                                        level = "level_3"
                                    elif prev_level == "level_3":
                                        level = "level_3"
                                    else:
                                        level = "no_hint"
                                elif prev_acc > max_threshold:
                                    if prev_level == "level_3":
                                        level = "level_2"
                                    elif prev_level == "level_2":
                                        level = "level_1"
                                    elif prev_level == "level_1":
                                        level = "no_hint"
                                    elif prev_level == "no_hint":
                                        level = "no_hint"
                                    else:
                                        level = "no_hint"

                            current_level_by_batch_idx[idx] = level
                            if level in level_to_indices:
                                level_to_indices[level].append(idx)

                        hint_payloads: dict[int, dict[str, Any]] = {}
                        hint_payloads_raw: dict[int, str] = {}
                        hint_failed = 0
                        hint_attempts = 0
                        generated_payloads: dict[int, dict[str, Any]] = {}

                        need_hint_indices = []
                        for level_key in ["level_1", "level_2", "level_3"]:
                            need_hint_indices.extend(level_to_indices[level_key])

                        if need_hint_indices:
                            requests = []
                            for idx in need_hint_indices:
                                question, solution = self._extract_question_and_solution(base_batch, base_gen_batch, idx)
                                if question and solution:
                                    requests.append((idx, question, solution))

                            if requests:
                                generated_payloads, hint_failed, hint_attempts, hint_payloads_raw = self._generate_hints_batch(
                                    requests
                                )
                                hint_payloads.update(generated_payloads)
                                metrics["hint/generate_failed"] = hint_failed
                                metrics["hint/generate_attempts"] = hint_attempts
                                metrics["hint/generate_requested"] = len(requests)
                                metrics["hint/generate_success"] = len(generated_payloads)

                                if index_array is not None and generated_payloads:
                                    for idx, payload in generated_payloads.items():
                                        self.reward_tracker.set_last_hint_payload(str(index_array[idx]), payload)
                                    self.reward_tracker.log_hint_raw(
                                        index_array,
                                        {idx: raw for idx, raw in hint_payloads_raw.items()},
                                        self.global_steps,
                                        used=False,
                                        failed=False,
                                    )
                                    self.reward_tracker.log_hint_payloads(
                                        index_array,
                                        generated_payloads,
                                        self.global_steps,
                                        used=False,
                                        failed=False,
                                    )

                                if index_array is not None and hint_failed:
                                    failed_indices = [idx for idx, _, _ in requests if idx not in generated_payloads]
                                    if failed_indices:
                                        fallback_used = 0
                                        for idx in failed_indices:
                                            fallback_payload = self.reward_tracker.get_last_hint_payload(
                                                str(index_array[idx])
                                            )
                                            if fallback_payload:
                                                hint_payloads[idx] = fallback_payload
                                                fallback_used += 1
                                        if fallback_used:
                                            metrics["hint/fallback_used"] = fallback_used
                                        failed_payloads = {idx: {} for idx in failed_indices}
                                        self.reward_tracker.log_hint_payloads(
                                            index_array,
                                            failed_payloads,
                                            self.global_steps,
                                            used=False,
                                            failed=True,
                                        )
                                        self.reward_tracker.log_hint_raw(
                                            index_array,
                                            {idx: hint_payloads_raw.get(idx, "") for idx in failed_indices},
                                            self.global_steps,
                                            used=False,
                                            failed=True,
                                        )

                        gen_batch_to_use = base_gen_batch
                        hint_applied_prompts: set[int] = set()
                        effective_level_by_batch_idx = {idx: "no_hint" for idx in range(len(base_batch))}

                        for level_key in ["level_1", "level_2", "level_3"]:
                            target_indices = level_to_indices[level_key]
                            if not target_indices:
                                continue
                            payload_subset: dict[int, dict[str, Any]] = {}
                            for idx in target_indices:
                                payload = hint_payloads.get(idx)
                                if payload:
                                    payload_subset[idx] = payload
                            if not payload_subset:
                                continue

                            hinted = self._apply_hints_to_gen_batch(
                                gen_batch_to_use, base_batch, target_indices, payload_subset, level_key
                            )
                            if hinted is None:
                                continue
                            gen_batch_to_use, applied_indices = hinted

                            if applied_indices:
                                hint_applied_prompts.update(applied_indices)
                                for idx in applied_indices:
                                    effective_level_by_batch_idx[idx] = level_key

                                if index_array is not None:
                                    payload_applied = {
                                        idx: payload_subset[idx] for idx in applied_indices if idx in payload_subset
                                    }
                                    if payload_applied:
                                        self.reward_tracker.log_hint_payloads(
                                            index_array,
                                            payload_applied,
                                            self.global_steps,
                                            used=True,
                                            failed=False,
                                        )
                                    raw_subset = {
                                        idx: hint_payloads_raw.get(idx, "")
                                        for idx in applied_indices
                                        if idx in hint_payloads_raw
                                    }
                                    if raw_subset:
                                        self.reward_tracker.log_hint_raw(
                                            index_array,
                                            raw_subset,
                                            self.global_steps,
                                            used=True,
                                            failed=False,
                                        )

                        batch, reward_tensor, reward_extra_infos_dict = run_rollout(gen_batch_to_use)

                        sequence_rewards = reward_tensor.sum(dim=-1) if reward_tensor is not None else None
                        rewards_per_question = (
                            sequence_rewards.reshape(len(base_batch), rollout_repeat)
                            if sequence_rewards is not None
                            else None
                        )

                        accuracies = []
                        if rewards_per_question is not None:
                            correct_mask = rewards_per_question > 0
                            accuracies = correct_mask.float().mean(dim=1).cpu().tolist()

                            level_accumulators = {"no_hint": [], "level_1": [], "level_2": [], "level_3": []}
                            for idx, acc in enumerate(accuracies):
                                level = effective_level_by_batch_idx.get(idx, "no_hint")
                                level_accumulators[level].append(acc)

                            metrics["hint/prompts_with_hint"] = len(hint_applied_prompts)
                            for level_key, accs in level_accumulators.items():
                                metrics[f"hint/used_{level_key}"] = len(accs)
                                if accs:
                                    metrics[f"hint/acc_{level_key}_mean"] = float(np.mean(accs))

                            if index_array is not None:
                                self.reward_tracker.log_hint_accuracy(
                                    index_array, effective_level_by_batch_idx, accuracies, self.global_steps
                                )
                                metrics["hint/accuracy_min_threshold"] = float(min_threshold)
                                metrics["hint/accuracy_max_threshold"] = float(max_threshold)
                                for idx, acc in enumerate(accuracies):
                                    index_str = str(index_array[idx])
                                    current_level = effective_level_by_batch_idx.get(idx, "no_hint")
                                    self.reward_tracker.set_hint_level(index_str, current_level)
                                    self.reward_tracker.set_last_hint_accuracy(index_str, acc)
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # Add reward tracking
                        self.reward_tracker.update(batch, self.global_steps)
                        zero_reward_stats = self.reward_tracker.get_zero_reward_stats()
                        metrics.update(zero_reward_stats)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
