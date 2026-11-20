# NVIDIA-NeMo/RL#4166 实验记录：FSDP2 / AutoModel 路径的 activation CPU offload

整理时间：2026-11-20。作者：杨晔塬（GitHub `shsaihdsaiudh`）。

## 背景

Issue：https://github.com/NVIDIA-NeMo/RL/issues/4166

Nemotron 3.5 Super 用 FSDP2/AutoModel 跑 GRPO（训练 16 张 GB200，TP/CP/EP = 1/1/1，每个 microbatch 32K token）时，在 MoE 前向（`relu2_deepep`）OOM。

查下来有两个原因：

1. **AC 覆盖不全（主因）**：`expert_parallel_size == 1` 时，Automodel 的 `NemotronHModelParallelizer` 只给 mamba 和 mlp block 加 activation checkpointing，attention 和 MoE block 的激活全部留在显存里。EP>1 时走 `moe.parallelizer.apply_ac`，每个 block 都会加。
2. **NeMo-RL 的 FSDP2 路径没有 activation offload**：开 AC 以后，每个 block 的输入还是要从前向留到反向。Megatron 路径有 `fine_grained_activation_offloading`，AutoModel 路径没有。

## 产出

| 内容 | 位置 | 状态（2026-11-20） |
|---|---|---|
| AC 覆盖修复 | https://github.com/NVIDIA-NeMo/Automodel/pull/4834 ，fork 分支 `shsaihdsaiudh/Automodel:shsaihdsaiudh/fix/nemotronh-ep1-full-ac` | 已开，等 NVIDIA 成员 `/ok to test` |
| activation CPU offload | https://github.com/NVIDIA-NeMo/RL/pull/5325 ，fork 分支 `shsaihdsaiudh/RL:issue-4166-activation-offload` | 已开，2 个 commit：`c6ebc7cd7`（功能）、`195a099cd`（加固）。CI 要 `/ok to test 195a099cd421be51f63fbb1ab19fbdbf05e219f4` |

PR #5325 的描述原文在 `notes/pr-5325-body.md`。

### PR #5325 做了什么

- 新配置 `policy.automodel_cfg.activation_offloading`（默认 `false`，必须同时开 `activation_checkpointing`，否则 `validate_and_prepare_config` 报 `ValueError`）。
- `nemo_rl/models/automodel/activation_offload.py`：`ActivationCpuOffload(model)`，基于 `saved_tensors_hooks`。
  - pack：≥1 MiB 的普通 CUDA 张量拷到 pinned host 内存（当前 stream，`non_blocking`），并对源张量 `record_stream`。
  - 不拷：张量子类（DTensor 等）、参数及其 view、和 model 参数共享 storage 的张量（`detach()`、`.data`、DTensor 参数的 local tensor、FSDP 前向时挂上的 unsharded 参数）。
  - 同一个张量被多个算子保存时只拷一份（按 `id` + 弱引用 + `_version` 去重）。
  - `empty_host_cache()`：同步后调用 `torch.accelerator.empty_host_cache()`，把缓存的 pinned 内存还给系统。
- `train.py`：只包训练时的 `model_forward`，logits、loss、logprob/参考模型/评估都不受影响。
- 两个 automodel worker 各建一个实例，每个训练 step 结束时释放 pinned 缓存。
- 文档：`docs/design-docs/training-backends.md` 的 "Activation CPU offload" 一节。

## 实验结果

### 1. Super 尺寸的单卡模拟（1×H20，`scripts/super_proxy_memory.py`，日志 `results/proxy/`）

32 experts top-22，32K token。

每层保留的显存（GiB）：

| 层 | 不开 AC | 开 AC |
|---|---|---|
| mamba | 2.13 | 0.25 |
| MoE | 10.34 | 0.26 |
| attention | 1.04 | 0.25 |

9 层 `MEMEMEM*E` stack：不开 AC 51.6 GiB，旧 AC 44.1 GiB，修复后 3.0 GiB；时间 2.27 / 2.50 / 3.01 s。

推算到 Super：旧 AC 约 432 GiB 激活（必然 OOM），修复后约 22.5 GiB（88 个 bf16 block 输入）。模型状态每个 rank 112.4 GiB（fp32 Adam），bf16 moments 时 84.3 GiB，显存预算约 175 GiB。head（真实词表）：不分块时前向/反向峰值 40/56 GiB，NeMo-RL 的 `ChunkedDistributedLogprob` chunk 2048 时 10/17.25 GiB。

offload，`E*E*E`，32K：保留显存 2.17 → 0.17 GiB，前向 +8%，反向 +2%，host pinned 2.01 GiB。

### 2. 梯度一致性

**结论：offload 不改变梯度。** 早期看到的"梯度不一致"（step-1 相对 L2 误差 0.648）来自每次重新构建模型再对比：不开 offload、只是重建两次模型，误差也有 0.03–0.88。正确的对比方法是同一个模型实例、同一个 batch，交替跑 plain 和 offload（`SAME_MODEL=1`）。

| 配置 | 结果 |
|---|---|
| 1×H20，单卡强制走 FSDP 策略（`FORCE_FSDP=1`），`MEM*EMEM*E` | 5 次逐位一致；开不开 deterministic 都一样；offload 放在第一个跑也一致 |
| 1×H20，deterministic，每次重建模型，纯 MoE / 纯 Mamba / 纯 attention 10 层，3 步 AdamW | loss 和梯度逐位一致 |
| 8×H20e，FSDP8，EP=1 | 8 张卡逐位一致（`results/cml-wlcb/t-20261120000143-ko4ym.stdout.log`） |
| 8×H20e，FSDP8，EP=8 + DeepEP | 8 张卡逐位一致（同上） |
| 第二个 commit 之后，1×H20 `FORCE_FSDP=1` | 仍逐位一致，host 拷贝仍是 13 个张量 |

两次构建的参数和 buffer 是逐位相同的（77 个张量）。跨实例的噪声很可能来自构建路径读到未初始化显存，见"遗留问题"。

### 3. 显存（同一个 Nemotron-H 模型，8×H20e，EP=1，每卡 16K token）

前向后保留 2.56 → 2.10 GiB，step 峰值 6.47 → 6.00 GiB，host pinned 0.47 GiB。拷到 host 的是 10 个 bf16 block 输入加 head 附近 3 个张量。`fsdp_output_dtype` 设 float32 或 bfloat16 结果相同：FSDP 在 block 保存输入前已经把输入转成 param dtype。

### 4. 8 卡 SFT（Qwen2.5-1.5B，8×H20e，AC 开，sequence packing 每 microbatch 32K token，20 步）

日志：`results/cml-wlcb/t-20261120000144-iaoj1.stdout.log`（先 off 后 on，同一台机器）。显存来自 nvidia-smi 每 2 秒采样，原始数据在 `wlcb/logs/sft_offload/sft-8g-1119-{off,on}/nvsmi.csv`，下表是 `scripts/cml/sft_compare.py` 的输出。

| | offload 关 | offload 开 |
|---|---|---|
| 显存峰值 rank 1–7 | 38,707 MiB | 36,115 MiB |
| 显存峰值 rank 0 | 81,955 MiB | 78,363 MiB |
| step 时间中位数（第 3 步起） | 11.04 s | 10.98 s |
| loss | 第 1 步 0.7572 | 第 1 步 0.7572，20 步最大差 0.0024 |

估算 block 输入 28 × 32768 × 1536 × 2 B ≈ 2.6 GiB，和实测少的 2.5 GiB 一致。没有跑两次 offload 关的对照，0.0024 的 loss 差是否全是正常波动没有确认。

### 5. 1 卡 SFT smoke（Qwen2.5-0.5B，3 步，offload 开，`results/dev/logs/sft_offload/`）

| 运行 | 代码 | loss |
|---|---|---|
| smoke1g-off | 第一版 | 1.63 / 0.84 / 0.49 |
| smoke1g-dedup | 加去重后 | 1.6316 / 0.8427 / 0.4959 |
| smoke1g-fix2 | 第二个 commit | 1.6316 / 0.8432 / 0.4961 |

### 6. 其它测量

- PyTorch 的 pinned 分配器按 2 的幂取整并缓存：申请 600 MiB 实际占 1024 MiB；接着申请 1100 MiB 再占 2048 MiB，前面的块不释放。
- `torch.accelerator.empty_host_cache()` 只释放拷贝已完成的块，调用前要 `torch.cuda.synchronize()`。
- 开发机 host 上 pin 4 GiB（256 MiB 一块）用 1.35 s，约 3 GiB/s；按这个速度 Super 的 22.5 GiB 每个 step 重新 pin 约 7 s。
- 参数 storage 扫描开销（600 个 DTensor 参数，`scripts/bench_param_scan.py`）：`to_local()` + `_cdata` 0.92 ms/次；`_local_tensor` + `data_ptr` 0.13 ms/次。实现里按参数对象缓存 storage key。
- 跨 stream 复用（`scripts/check_record_stream.py`）：不加 `record_stream`，源张量的显存会在拷贝排队时被分配它的 stream 立刻重新拿去用；加了以后不会。

## 和开源实现的对比（2026-11-20）

对比了 torchtune `OffloadActivations`、PrimeRL `act_offloading.py`、verl `activation_offload.py`（来自 TE cpu_offload）、Megatron-LM fine-grained activation offloading。

已在 PR #5325 第二个 commit 修掉：参数别名漏判（PrimeRL 修过同一个问题）、缺 `record_stream`、pinned 缓存不释放。

还没做，可以作为后续 PR：

1. **stream 重叠和预取**：现在 D2H/H2D 和计算串行。verl/TE/Megatron 用独立 d2h/h2d stream，按层分组，前向边算边拷、反向提前拷回。torchtune 也用 stream，但 PrimeRL 因为内存泄漏和 loss 对不上把 stream 关了，要小心。估算 Super 每 block 256 MiB：H20 PCIe 单程约 5 ms，88 层约 0.45 s；GB200 C2C 88 层约 50 ms。
2. **最后一层留在 GPU**：反向最先用到最后一层的输入和 head 附近的张量。verl 不 offload 最后一层，Megatron 保留每类模块的最后一组，torchtune 跳过 output head。要可靠识别 block 边界（checkpoint wrapper 或 FSDP 模块），按张量个数判断会因模型而异。估算收益不到一个 step 的 1%。
3. **pinned 缓冲池**：Megatron 有按 (shape, dtype) 复用的 tensor pool 和固定大小 buffer，能避开 2 的幂取整和每步重新 pin。
4. **部分 offload 开关**：Megatron 的 `activation_offload_fraction`。
5. **可观测性**：offload 字节数指标、host 内存使用率告警（torchtune 60%，PrimeRL 90%）。
6. **共享拷贝的 GPU 复用**：同一份 host 拷贝被多个使用者 unpack 时各拷回一次；PrimeRL 会共享拷回的 GPU 张量。去重也只认同一个 Python 对象，view/别名会拷两份（实测 FSDP 路径没有重复）。

## 遗留问题（和 offload 无关，未处理）

1. 开 `torch.utils.deterministic.fill_uninitialized_memory` 后，r0.6 `from_config` 构建的 Nemotron-H 在 layer 0 的 q/k/v 和 Mamba `in_proj` 输出出现 NaN（输入没有 NaN）。换 `ATTN=sdpa`、`LINEAR=torch` 也一样，纯 MoE 模型不出现。说明前向读到了未初始化显存，很可能是跨实例梯度噪声的来源。单卡强制 FSDP 的 hack 下复现，多卡未确认。
2. 8 卡 EP=8 + DeepEP 时，每次新建模型训练，不开 offload 的 loss 也是 NaN，只有第一次构建（warmup）正常。
3. 随机初始化的 Mamba 会出 NaN 梯度（和 offload 无关）。
4. Automodel 在单卡时跳过 FSDP 并行策略，走另一套 AC（只包 mixer），和多卡生产路径不同。

## 复现方法

脚本里的路径是公司开发环境的路径，换环境时要改。

- **NeMo-RL 代码**：fork 分支 `issue-4166-activation-offload`（基于 upstream `e6449ae6a`）。Automodel 子模块是 `b916107a`（r0.6.0），实验时打了 `patches/automodel-r0.6-nemotronh-ep1-full-ac.patch`（和 Automodel PR #4834 等价，r0.6 上的版本），这个补丁不在 NeMo-RL 的 PR 里。
- **单元测试**：`pytest tests/unit/models/automodel/test_activation_offload.py`（GPU，不要加 `--automodel-only`，否则收集到 0 个）；`pytest --automodel-only tests/unit/models/automodel/test_automodel_setup.py tests/unit/models/automodel/test_automodel_train.py`。
- **多卡检查**：`scripts/multi_gpu_offload_check.py`，`torchrun --nproc-per-node N`。主要环境变量：
  - `SAME_MODEL=1`：同一实例交替对比（判断正确性用这个）；`SAME_ORDER` 改顺序。
  - `RECORD=1`：打印拷到 host 的张量。
  - `EP_SIZES`、`SEQ`、`STEPS`、`PATTERN`（`M`=mamba，`E`=moe，`*`=attention，`-`=mlp）、`OUTPUT_DTYPE`、`ATTN`、`LINEAR`。
  - `FORCE_FSDP=1`：单卡也走多卡的 FSDP 策略。
  - `DETERMINISTIC=1`（要 `CUBLAS_WORKSPACE_CONFIG=:4096:8`），`FILL_UNINIT=0/1`，`NAN_TRACE=1`，`BUILD_CHECK=1`。
  - `clone`/`verify`/`DEBUG`/`sync`/`keepref` 模式是早期排查用的，还按旧 API 写，没有更新。
  - 单卡封装 `scripts/cml/run_multi_gpu_check.sh`，批量跑 `scripts/cml/sweep_1gpu.sh` + `summarize.py`。
- **SFT**：`scripts/cml/sft_offload_entry.sh`（`MODEL`、`STEPS`、`NGPU`、`MB_TOKENS`、`OFFLOAD`），一次跑 off/on 两组用 `sft_pair_entry.sh`，对比用 `sft_compare.py`。8 卡入口 `mgc_entry.sh`。
- `scripts/cml/jobs/` 是提交集群任务用的 YAML，内部地址已经替换成 `<redacted>`。

## 目录

```
README.md                  本文
notes/pr-5325-body.md      PR #5325 描述
patches/                   实验时 Automodel 子模块的本地补丁和基线 commit
scripts/                   实验脚本（scripts/cml/ 是集群和批量运行的封装）
results/proxy/             Super 尺寸单卡模拟日志
results/dev/               开发机日志（dense AC 检查、SFT smoke）
results/cml-wlcb/          8 卡任务的 stdout
wlcb/logs/                 集群共享存储（乌兰察布）上的 8 卡任务日志、sysinfo、code.md5、nvsmi.csv
EXCLUDED-wlcb.txt          没收进来的文件（这次为空）
```

## 脱敏

这个分支在公开 fork 上。内网 IP、主机名、内部域名和镜像地址、集群卷名、队列和资源名都替换成了 `<redacted…>`，日志里的环境变量转储已删除。路径 `/planning-users/yangyeyuan/...` 保留，用来对照原始位置。
