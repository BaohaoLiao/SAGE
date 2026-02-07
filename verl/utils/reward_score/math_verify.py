# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import warnings
from functools import partial
from typing import Optional, Tuple

import multiprocessing as mp
import numpy as np
import signal
from contextlib import contextmanager

try:
    # TimeoutException 未直接使用，这里仍保留兼容；如不需要可删除
    from math_verify.errors import TimeoutException  # noqa: F401
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# -------------------- 进程内硬超时（仅 POSIX） --------------------
@contextmanager
def _time_limit(seconds: int):
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError("compute_score timed out")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# -------------------- 解析 boxed --------------------
def extract_boxed_answer(s: str) -> Optional[str]:
    """提取最后一个 \\boxed{...} 的内容；未找到返回 None。支持花括号嵌套。"""
    needle = r"\boxed{"
    start = s.rfind(needle)
    if start == -1:
        return None
    i = start + len(needle)
    depth = 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[start + len(needle):i].strip()
        i += 1
    return None

# -------------------- 评分（worker 进程内复用 verify_func） --------------------
_VERIFY_FUNC = None  # 每个 worker 进程持有一份

def _ensure_verify_func():
    global _VERIFY_FUNC
    if _VERIFY_FUNC is None:
        _VERIFY_FUNC = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _VERIFY_FUNC

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0.0) -> float:
    verify_func = _ensure_verify_func()
    mo = extract_boxed_answer(model_output)
    if not mo:
        return timeout_score
    gt_boxed = "\\boxed{" + ground_truth + "}"
    pred_boxed = "\\boxed{" + mo + "}"
    try:
        score, _ = verify_func([gt_boxed], [pred_boxed])
        return float(score)
    except Exception:
        return timeout_score

# -------------------- worker：在进程内做硬超时保护 --------------------
def _evaluation_worker(
    model_output: str,
    ground_truth: str,
    evaluation_func,
    timeout_score: float,
    inner_timeout_s: int,
) -> float:
    try:
        with _time_limit(inner_timeout_s):
            return float(evaluation_func(model_output, ground_truth, timeout_score=timeout_score))
    except Exception as e:
        # 包括 TimeoutError 在内，统一按超时分处理
        warnings.warn(f"worker error: {e}")
        return float(timeout_score)

# -------------------- 并行入口（接口保持不变） --------------------
def parallel_compute_score(
    model_outputs: list,
    ground_truths: list,
    timeout: int = 20,
    num_processes: int = 32,
    *,
    timeout_score: float = 0.0,          # 新增可选参数：超时/异常的默认分（默认与 compute_score 一致）
    maxtasksperchild: int = 200,         # 每个子进程最多处理多少任务后自动重启（防止状态污染）
    start_method_priority: tuple = ("forkserver", "spawn"),  # 优先使用更稳的启动方式
) -> list:
    """
    和你原来的接口/用法兼容；内部改为：
      - worker 内部 signal.alarm 做硬超时（inner_timeout_s = timeout-1）
      - 用 get_context(...) 创建 Pool（forkserver/spawn），不使用 with Pool
      - 超时样本直接记为 timeout_score，并重启池处理剩余任务
    """
    assert len(model_outputs) == len(ground_truths), "model_outputs 与 ground_truths 长度需一致"
    n = len(model_outputs)
    if n == 0:
        return []

    # 选择 mp 上下文（不改全局 start method）
    ctx = None
    for m in start_method_priority:
        try:
            ctx = mp.get_context(m)
            break
        except ValueError:
            continue
    if ctx is None:
        ctx = mp.get_context()  # fallback

    inner_timeout_s = max(1, int(timeout) - 1)

    tasks_to_process = list(enumerate(zip(model_outputs, ground_truths)))
    results = [-1.0] * n
    worker_func = partial(
        _evaluation_worker,
        evaluation_func=compute_score,
        timeout_score=timeout_score,
        inner_timeout_s=inner_timeout_s,
    )

    # 主循环：直到所有结果填满
    while any(r == -1.0 for r in results):
        remaining_tasks: list[Tuple[int, Tuple[str, str]]] = [
            (i, args) for i, args in tasks_to_process if results[i] == -1.0
        ]
        if not remaining_tasks:
            break

        # 创建池（不使用 with，便于我们在超时/异常时立即 terminate）
        pool = ctx.Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)
        need_restart = False
        try:
            # 提交任务
            async_results = {
                idx: pool.apply_async(worker_func, args=task_args)
                for idx, task_args in remaining_tasks
            }

            # 逐个取回；一旦某个任务外层 get 超时，标记重启（同时给该样本 timeout_score）
            for idx, res in async_results.items():
                try:
                    score = res.get(timeout=timeout)
                    results[idx] = float(score)
                except ctx.TimeoutError:
                    warnings.warn(f"Timeout on index {idx}; will restart pool. Marking timeout_score.")
                    results[idx] = float(timeout_score)  # ✅ 直接给分，避免下轮再卡
                    need_restart = True
                    break
                except Exception as e:
                    warnings.warn(f"Error retrieving result at index {idx}: {e}; set timeout_score.")
                    results[idx] = float(timeout_score)

            # 如果中途出现超时，终止池子，下一轮只处理剩余样本
            if need_restart:
                try:
                    pool.terminate()
                finally:
                    pool.join()
                continue

            # 正常收尾
            pool.close()
            pool.join()

        finally:
            # 双保险：任何异常路径都尽量释放进程
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    # （可选）输出简单统计
    try:
        print("Length of results:", len(results))
        print("Mean score original:", float(np.mean(results)))
        print("Max/Min score original:", float(np.max(results)), float(np.min(results)))
    except Exception:
        pass

    return results
