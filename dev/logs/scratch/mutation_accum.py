import pytest
import torch

import nemo_rl.distributed.fused_linear_logprobs as fl
from tests.unit.distributed import test_fused_linear_logprobs as t


def bf16_accumulate(acc, mat1, mat2):
    acc.copy_((acc + (mat1 @ mat2).float()).bfloat16().float())


mp = pytest.MonkeyPatch()
mp.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
mp.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
mp.setattr(fl, "_accumulate_fp32", bf16_accumulate)
try:
    t.test_weight_grad_is_as_accurate_as_one_unfused_matmul()
    print("MUTANT SURVIVED")
except AssertionError:
    print("mutant killed")
