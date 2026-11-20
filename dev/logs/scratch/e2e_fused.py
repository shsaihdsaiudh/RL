"""End-to-end check of policy.automodel_cfg.use_fused_linear_logprobs through the Ray
AutoModel policy worker: get_logprobs, then one GRPO-loss train step.

Usage: python e2e_fused.py {fused,unfused} OUT.pt [MODEL] [SEQ]
"""

import sys

import ray
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.models.policy.test_automodel_worker import create_test_config

mode, out_path = sys.argv[1], sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-0.5B"
seq = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
fused = mode == "fused"
batch = 2

config = create_test_config(model_name=model_name, precision="bfloat16")
config["generation"]["temperature"] = 0.7
config["train_global_batch_size"] = batch
config["train_micro_batch_size"] = batch
config["logprob_batch_size"] = batch
config["dynamic_batching"]["enabled"] = False
config["automodel_cfg"]["lora_cfg"]["enabled"] = False
config["automodel_cfg"]["use_fused_linear_logprobs"] = fused
config["automodel_cfg"]["fused_linear_logprobs_chunk_size"] = 1024
config["max_total_sequence_length"] = seq
tokenizer = get_tokenizer(config["tokenizer"])
config["generation"] = configure_generation_config(config["generation"], tokenizer)

init_ray()
cluster = RayVirtualCluster(
    name=f"fused-{mode}",
    bundle_ct_per_node_list=[1],
    use_gpus=True,
    num_gpus_per_node=1,
    max_colocated_worker_groups=1,
)
policy = Policy(
    tokenizer=tokenizer,
    config=config,
    init_optimizer=True,
    init_reference_model=False,
    cluster=cluster,
    name_prefix=f"fused_{mode}",
)
try:
    torch.manual_seed(0)
    input_ids = torch.randint(0, min(len(tokenizer), 150000), (batch, seq))
    input_lengths = torch.tensor([seq, seq - 517])
    positions = torch.arange(seq).unsqueeze(0)
    valid = (positions < input_lengths.unsqueeze(1)).float()
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": valid,
            "sample_mask": torch.ones(batch),
        }
    )
    policy.prepare_for_lp_inference()
    logprobs = policy.get_logprobs(data)["logprobs"]
    info_lp = ray.get(policy.worker_group.workers[0].get_gpu_info.remote())

    data["prev_logprobs"] = logprobs
    data["generation_logprobs"] = logprobs
    data["reference_policy_logprobs"] = logprobs
    data["advantages"] = torch.randn(batch, seq) * valid
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(), use_fused_linear_logprobs=fused)
    policy.prepare_for_training()
    results = policy.train(data, loss_fn)
    info_train = ray.get(policy.worker_group.workers[0].get_gpu_info.remote())
    out = {
        "logprobs": logprobs,
        "loss": results["loss"],
        "grad_norm": results["grad_norm"],
        "peak_after_logprobs_mb": info_lp["peak_memory_allocated_mb"],
        "peak_after_train_mb": info_train["peak_memory_allocated_mb"],
    }
    torch.save(out, out_path)
    print(
        f"[e2e] {mode}: loss={results['loss']} grad_norm={results['grad_norm']} "
        f"peak_after_logprobs={out['peak_after_logprobs_mb']:.0f}MB "
        f"peak_after_train={out['peak_after_train_mb']:.0f}MB"
    )
finally:
    policy.shutdown()
    cluster.shutdown()
