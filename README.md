# SAGE: Self-Hinting Language Models for RLHF

SAGE (Self-hint Aligned GRPO with Privileged Supervision) is a reinforcement-learning recipe that injects **privileged hints during training only** to keep Group Relative Policy Optimization (GRPO) updates informative under sparse terminal rewards. The approach is described in `paper/SAGE.pdf` and implemented here on top of the VERL RLHF stack.

## What the paper shows
- GRPO groups often collapse (all rewards 0/1), killing advantages; hints reshape rollouts without changing the verifier reward.
- For each prompt *x*, the model samples a compact hint *h* (plan/decomposition) and then rolls out τ∼πθ(·|x,h); at test time *h*=∅.
- Two variants: **SAGE** (on-policy, searches hint levels until a non-degenerate group appears) and **SAGE-LIGHT** (lightweight per-prompt scheduler that raises hint level when accuracy stays low).
- On math/logic benchmarks (AIME24/25, AMC23, MATH-500, Minerva, Olympiad, GPQA, MMLU-Pro) SAGE boosts average accuracy by ~+5–11 points over GRPO across Llama-3.2-3B, Qwen2.5-7B, and Qwen3-4B models. Training with hints roughly costs 2.3× (SAGE) or 1.2× (SAGE-LIGHT) the GRPO wall-clock (Table 3).
- Hard-prompt study (4.5k items) uses 32 rollouts per prompt/step; hints of any level sharply raise accuracy, with online self-hinting best across levels (Fig. 3).

## Repository layout
- `paper/SAGE.pdf` — full paper.
- `recipe/hint/main_hint.py` — Hydra entrypoint for PPO/GRPO training with hinting.
- `recipe/hint/hint_trainer.py` — SAGE/SAGE-LIGHT trainer built on VERL; defines hint prompts and rollout logic.
- `recipe/hint/config/hint_trainer.yaml` — minimal config that toggles hint mode and thresholds.
- `recipe/hint/reward_tracker.py` — tracks per-prompt rewards, hint levels, and accuracies.
- `verl/` — VERL RLHF/GRPO engine (actors, critics, configs, workers, utils).

## Requirements (typical)
- Python ≥3.10, CUDA GPUs with NCCL.
- PyTorch, Ray, Hydra/OmegaConf, Transformers, VLLM/FlashAttention (as used by VERL), sentencepiece, datasets/accelerate.
- Set `PYTHONPATH=$(pwd):$PYTHONPATH` so `recipe` and `verl` import correctly.

## Quick start
1) Install deps in a new environment (adapt versions to your hardware):
```bash
pip install torch ray[default] hydra-core omegaconf transformers datasets accelerate sentencepiece vllm flash-attn
```
2) Prepare training/validation data in parquet with at least a `prompt` column (see `verl/trainer/config/data/legacy_data.yaml` for expected fields; `solution` is consumed when generating hints).

3) Run SAGE-LIGHT (default lightweight scheduler):
```bash
python recipe/hint/main_hint.py \
  trainer.method=sage-light \
  data.train_files=/path/train.parquet data.val_files=/path/val.parquet \
  actor_rollout_ref.model.pretrained_model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
  trainer.nnodes=1 trainer.n_gpus_per_node=8
```

4) Run full SAGE (on-policy hint search):
```bash
python recipe/hint/main_hint.py \
  trainer.method=sage \
  data.train_files=/path/train.parquet data.val_files=/path/val.parquet \
  actor_rollout_ref.model.pretrained_model_name_or_path=Llama-3.2-3B-Instruct \
  trainer.nnodes=1 trainer.n_gpus_per_node=8
```
Key knobs (override via Hydra):
- `trainer.hint_accuracy_min_threshold` / `trainer.hint_accuracy_max_threshold` — thresholds for stepping hint levels (LIGHT).
- `actor_rollout_ref.rollout.response_length` and `data.max_response_length` — rollout budget (8192 for Table 1; 2048 elsewhere per Appendix C.2).
- `actor_rollout_ref.rollout.num_generations_per_prompt` — group size (*G*); use larger values (e.g., 32) for very hard prompts as in Fig. 3.

5) Evaluation: reuse the rollout config with `trainer.eval_only=true` and point `data.val_files` to the benchmark parquet; sampling uses temperature 0.6 and top-p 0.95 in the paper.

## How hinting is wired in code
- Hint prompts live in `HINT_SYSTEM_PROMPT` and `HINT_USER_PROMPT_TEMPLATE` inside `recipe/hint/hint_trainer.py` and produce three levels of procedural hints without revealing answers.
- During training, hint payloads and accuracies are logged via `RewardTracker` for per-prompt scheduling and analysis.
- Deployment sets hint level to 0 (no hint) so the production policy matches standard GRPO inference.

## Citation
If you use SAGE, please cite the paper:
```
@article{liao2025sage,
  title  = {Self-Hinting Language Models Enhance Reinforcement Learning},
  author = {Baohao Liao and Hanze Dong and Xinxing Xu and Christof Monz and Jiang Bian},
  year   = {2025}
}
```

## License
Apache License 2.0. See `LICENSE` for details.

