"""The cross-stream test body, with and without record_stream in the host copy."""

import torch


def copy_to_host(tensor, record):
    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
    stream = torch.cuda.current_stream(tensor.device)
    host.copy_(tensor, non_blocking=True)
    if record:
        tensor.record_stream(stream)
    return host


for record in (False, True):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        source = torch.randn(1024, 1024, device="cuda")
    torch.cuda.current_stream().wait_stream(side)
    expected = source.clone()
    ptr = source.data_ptr()
    torch.cuda._sleep(100_000_000)
    host = copy_to_host(source, record)
    del source
    with torch.cuda.stream(side):
        reused = torch.zeros(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"record_stream={record}: reused_same_memory={reused.data_ptr() == ptr} host_copy_correct={torch.equal(host.cuda(), expected)}")
    del reused, host, expected
    torch.cuda.synchronize()
