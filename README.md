# NVIDIA-NeMo/RL#3976 实验记录：Colocated training state offload is inefficient

Issue：https://github.com/NVIDIA-NeMo/RL/issues/3976

这个工作区做出了三个 PR（都在 fork `shsaihdsaiudh/RL`）：

| PR | 分支 | 内容 |
|---|---|---|
| https://github.com/NVIDIA-NeMo/RL/pull/5311 | `perf/megatron-optimizer-offload` | 用可复用的 pinned buffer offload Megatron 优化器状态 |
| https://github.com/NVIDIA-NeMo/RL/pull/5303 | `perf/megatron-gc-freeze` | 冻结 worker setup 后存活的对象（`gc_probe` 实验） |
| https://github.com/NVIDIA-NeMo/RL/pull/5327 | `fix/numa-bind-cpuset` | CPU pinning 失败时保留 NUMA 内存策略（`numa_bench` 实验） |

- 环境搭建说明：`local/env/README.md`（`bootstrap.sh`、`env.sh`、`nrl-uv-sync.sh`）。
- `local/probe-3976.patch`：计时探针补丁，不在任何 PR 里。
- 本地还有一个只在本地的分支 `tmp/3976-gc-probe`，没有推送。
- 8 卡结果主要在 `wlcb/logs/offload_8g/` 和 `wlcb/logs/gc_probe/`；`dev/logs/` 是开发机单卡和早期 8 卡运行。

## 目录

| 目录 | 来源 |
|---|---|
| `local/` | WSL 本地 `~/nemo-rl-3976/{env,logs,nemo_rl,probe-3976.patch}` |
| `dev/` | 开发机 `/planning-users/yangyeyuan/nemo-rl-3976/{logs,env}（不含 env/tools）` |
| `wlcb/` | 集群共享存储（乌兰察布）`/planning-users/yangyeyuan/nemo-rl-3976/{logs,env}（不含 env/tools）` |

`EXCLUDED-<目录>.txt` 列出没有收进来的文件及大小。收集时跳过的内容：Python/CUDA 环境、venv、wheel、模型和 checkpoint、`*.sqlite`（可以从 `*.nsys-rep` 重新导出）、NeMo-RL 的 rollout 样本 `train_data_step*.jsonl`、50 MB 以上的文件、`__pycache__`。

## 脱敏

这个分支在公开 fork 上。内网 IP、主机名、内部域名和镜像地址、集群卷名、队列和资源名都替换成了 `<redacted…>`；日志里的环境变量转储（`KEY=value` 行和 `env.txt`）已删除。`*.nsys-rep` 会把进程的环境变量写进报告（3976 的 trace 里有云服务 access key），所以全部没有收，清单在 `EXCLUDED-nsys-rep.txt`（如果有）。路径 `/planning-users/yangyeyuan/...` 保留，用来对照原始位置。

整理时间：2026-11-20。
