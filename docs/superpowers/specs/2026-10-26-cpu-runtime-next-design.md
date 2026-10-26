# CPU 运行时下一刀

日期：2026-10-26  
项目：cllm_inference  
前提：Linux x86、CPU、内存够、≤30B、单用户、零依赖。  
分支：`feat/cpu-runtime-next`

## 1. 目标

在已闭环的「能聊」之上补齐：可控上下文、真实 Q4_0/Q5_0 吞吐、可复现采样、最小 stop、decode attention SIMD、以及能测到 batched prefill 的 bench。

成功标准：

- `--ctx N` 把硬帽和 `config().n_ctx` 设为 `min(模型 n_ctx, N)`；未设则硬帽仍是模型 `n_ctx`。
- KV `reserve` 最多按 4096 token，不再按 32k–40k 盲预留。
- mmap 成功后调用 `madvise(..., MADV_WILLNEED)`；失败忽略。
- Q4_0 / Q5_0 在 `__AVX2__` 下走 SIMD `vec_dot`，容差与现网非 K 一致。
- CLI `--seed` / HTTP `seed` 可复现；缺省仍 `random_device`。
- CLI `--stop`（可重复）与 HTTP `stop`（字符串或数组）对已生成文本做后缀匹配，命中则截掉后缀并停。
- `attention()` 的 `head_dim` 内循环在 AVX2（及 NEON）下向量化。
- `model_bench` 的 prefill 段调用 `prefill()`，不再逐步 `forward()`。

## 2. 非目标

- Attention 按 head 并行
- AVX-VNNI / AVX512-VNNI 内核
- 进程内 `sched_setaffinity`
- `mlock` / `MAP_POPULATE`
- `repeat_penalty`、完整 Jinja、30B 模型下载
