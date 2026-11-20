# NVIDIA-NeMo/RL#4104 实验记录：nccl_reshard refit 支持生成侧 pipeline parallelism

Issue：https://github.com/NVIDIA-NeMo/RL/issues/4104

还没开始实现：需要 2 卡及以上，当时的开发机只有 1 张卡。这里只有环境搭建日志。本地仓库在 `main` `bfe65cdf6`，没有改动。

## 目录

| 目录 | 来源 |
|---|---|
| `dev/` | 开发机 `/planning-users/yangyeyuan/nemo-rl-4104/logs` |

`EXCLUDED-<目录>.txt` 列出没有收进来的文件及大小。收集时跳过的内容：Python/CUDA 环境、venv、wheel、模型和 checkpoint、`*.sqlite`（可以从 `*.nsys-rep` 重新导出）、NeMo-RL 的 rollout 样本 `train_data_step*.jsonl`、50 MB 以上的文件、`__pycache__`。

## 脱敏

这个分支在公开 fork 上。内网 IP、主机名、内部域名和镜像地址、集群卷名、队列和资源名都替换成了 `<redacted…>`；日志里的环境变量转储（`KEY=value` 行和 `env.txt`）已删除。`*.nsys-rep` 会把进程的环境变量写进报告（3976 的 trace 里有云服务 access key），所以全部没有收，清单在 `EXCLUDED-nsys-rep.txt`（如果有）。路径 `/planning-users/yangyeyuan/...` 保留，用来对照原始位置。

整理时间：2026-11-20。
