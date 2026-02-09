<div align="center">
<img alt="logo" src="assets/sage.png" width="400">

<div align="center">

## Self-Hinting Language Models Enhance Reinforcement Learning

When signal is lost for hard prompts during RL (all trajectories are wrong), the LLM self-generates a hint to help sampling, improving both prompt usage and LLM performance.

</div>

<p align="center">
  <a href="https://www.arxiv.org/abs/2602.03143"><img src="https://img.shields.io/badge/arXiv-2602.03143-b31b1b?style=flat&labelColor=555" alt="arXiv"></a>
  <a href="https://github.com/BaohaoLiao/SAGE"><img src="https://img.shields.io/badge/github-SAGE-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/collections/baohao/sage"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="huggingface"></a>
  <a href="https://huggingface.co/collections/baohao/sage"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue" alt="huggingface"></a>
  <a href="assets/sage_slide.mp4"><img src="https://img.shields.io/badge/▶ Video-English-red" alt="huggingface"></a>
<a href="assets/sage_slide_chinese.mp4"><img src="https://img.shields.io/badge/▶ Video-Chinese-red" alt="huggingface"></a>
</p>

</div>

## 🔥 News

- **[02/08/2026]** SAGE reproduction code is released! A [huggingface collection](https://huggingface.co/collections/baohao/sage) of datasets and models is also released.
- **[02/03/2026]** SAGE paper is released on [arXiv](https://www.arxiv.org/abs/2602.03143)!


## 🌟 Overview

<div align="center">
<img alt="logo" src="assets/overview.png" width="500">
</div>

When an LLM can’t sample any correct trajectory for a hard prompt, the LLM self-generates a hint from the reference solution for the prompt. The hint is then used together with the difficult prompt as input to the LLM, avoiding advantage collapse and ensuring the sampling of correct trajectories to update the policy model.

<div align="center">
<img alt="logo" src="assets/prompt_usage.png" width="400">
</div>
Without a hint, some hard prompts are never used for GRPO, while SAGE increases the prompt usage rate by 10% for the weaker LLM.

<div align="center">
<img alt="logo" src="assets/results.png" width="800">
</div>

The usage of hard prompts during RL encourages LLM's exploration, leading to consistently better performance.

<div align="center">
<img alt="logo" src="assets/training_dynamics.png" width="800">
</div>

Among all methods, SAGE retains the on-policy property of GRPO, having a similar entropy scale. And the learning from hard prompts promotes exploration, with the response length growing steadily for various LLMs.


## 📦 Installation
Our code is based on verl. If you already have a verl environment, you can use it and install the extra packages when prompted.
1. Create a new environment
    ```bash
    python -m venv ~/.python/sage
    source ~/.python/sage/bin/activate

    # Or use conda
    # conda create -n sage python==3.10
    # conda activate sage
    ```
2. Install dependencies
    ```bash
    pip install --upgrade pip
    pip install uv

    python -m uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    python -m uv pip install -U pip setuptools wheel packaging psutil
    python -m uv pip install flash-attn==2.8.0.post2 --no-build-isolation

    git clone https://github.com/BaohaoLiao/SAGE.git
    cd ./SAGE
    python -m uv pip install -r requirements.txt
    python -m uv pip install -e .
    python -m uv pip install vllm==0.10.1
    ```

## ⚡ Training
1. Prepare training set

    ```bash
    bash scripts/prepare_data.sh
    ```

2. Train with SAGE / SAGE-light. The key code is located in `recipe/hint`.

    ```bash
    bash scripts/run_sage.sh
    ```

3. Baselines (optional)
    - GRPO:
        ```bash
        bash scripts/run_grpo.sh
        ```
    - LUFFY: We use [LUFFY's open-sourced code](https://github.com/ElliottYan/LUFFY). The [training set](https://huggingface.co/datasets/baohao/luffy_train) is already preprocessed to LUFFY's style.
    - SFT: We use [LUFFY's open-sourced code](https://github.com/ElliottYan/LUFFY) for SFT. The [training set](https://huggingface.co/datasets/baohao/luffy_train) is already preprocessed to LUFFY's style.
    - Scaf-GRPO: We use [Scaf-GRPO's open-sourced code](https://github.com/JIA-Lab-research/Scaf-GRPO). The [training set](https://huggingface.co/datasets/baohao/scaf-grpo_train) is already preprocessed to Scaf-GRPO's style.

## 🤗 Trained Models

| Model name | Link |
| --- | --- |
| SAGE_Llama_3.2-3B-Instruct | https://huggingface.co/baohao/SAGE_Llama-3.2-3B-Instruct |
| SAGE-light_Llama-3.2-3B-Instruct | https://huggingface.co/baohao/SAGE-light_Llama-3.2-3B-Instruct  |
| SAGE_Qwen2.5-7B-Instruct | https://huggingface.co/baohao/SAGE_Qwen2.5-7B-Instruct |
| SAGE-light_Qwen2.5-7B-Instruct | https://huggingface.co/baohao/SAGE-light_Qwen2.5-7B-Instruct |
| SAGE_Qwen3-4B-Instruct-2507 | https://huggingface.co/baohao/SAGE_Qwen3-4B-Instruct-2507 |
| SAGE-light_Qwen3-4B-Instruct-2507 | https://huggingface.co/baohao/SAGE-light_Qwen3-4B-Instruct-2507 |

## 🎓 Evaluation
```bash
bash scripts/eval.sh
```

## 📝 Citation

If you find SAGE useful, please cite as:

```bibtex
@misc{liao2026selfhintinglanguagemodelsenhance,
      title={Self-Hinting Language Models Enhance Reinforcement Learning}, 
      author={Baohao Liao and Hanze Dong and Xinxing Xu and Christof Monz and Jiang Bian},
      year={2026},
      eprint={2602.03143},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.03143}, 
}
```

## 🙏 Acknowledgments
Our code is based on [verl](https://github.com/verl-project/verl) for training, [vllm](https://github.com/vllm-project/vllm) for sampling, and [oat](https://github.com/sail-sg/oat) for response grader. We really appreciate their contributions to the RL community.
