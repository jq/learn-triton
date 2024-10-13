import torch
import triton
import triton.language as tl

from typing import Optional
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def e4m3_to_bf16_torch(
        fp8_tensor: torch.Tensor, lookup_table: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:

    assert fp8_tensor.dtype == torch.uint8
    if out is not None:
        assert out.dtype == torch.bfloat16
        assert out.shape == fp8_tensor.shape
        return out.copy_(lookup_table[fp8_tensor.int()])
    return lookup_table[fp8_tensor.int()]

@triton.jit
def e4m3_to_bf16_triton_kernel(
        fp8_ptr,           # Pointer to input fp8 tensor
        lookup_ptr,        # Pointer to lookup table
        out_ptr,           # Pointer to output bf16 tensor
        n_elements,        # Number of elements
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    fp8_indices = tl.load(fp8_ptr + offsets, mask=mask, other=0).to(tl.int32)
    bf16 = tl.load(lookup_ptr + fp8_indices, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, bf16, mask=mask)

def e4m3_to_bf16_triton(
        fp8_tensor: torch.Tensor, lookup_table: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert fp8_tensor.dtype == torch.uint8
    if out is not None:
        assert out.dtype == torch.bfloat16
        assert out.shape == fp8_tensor.shape
    else:
        out = torch.empty_like(fp8_tensor, dtype=torch.bfloat16)

    n_elements = fp8_tensor.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    e4m3_to_bf16_triton_kernel[grid](fp8_tensor, lookup_table, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

def compare():
    e4m3_to_bf16_tensor = torch.load("data/e4m3_to_bf16.pt", weights_only=True).to('cuda')
    torch.manual_seed(0)
    size = 10
    fp8 = torch.randint(0, 256, (size,), device='cuda', dtype=torch.uint8)

    output_torch = e4m3_to_bf16_torch(fp8, e4m3_to_bf16_tensor)
    print("Output (Torch):", output_torch)
    output_triton = e4m3_to_bf16_triton(fp8, e4m3_to_bf16_tensor)
    print("Output (Triton):", output_triton)
    # max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    # print(f'The maximum difference between torch and triton is {max_diff}')

    # assert torch.allclose(output_torch, output_triton, atol=1e-3), "Outputs do not match!"

compare()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='e4m3_to_bf16-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    e4m3_to_bf16_tensor = torch.load("data/e4m3_to_bf16.pt", weights_only=True).to('cuda')
    fp8 = torch.randint(0, 256, (size,), device='cuda', dtype=torch.uint8)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        func = lambda: e4m3_to_bf16_torch(fp8, e4m3_to_bf16_tensor)
    elif provider == 'triton':
        func = lambda: e4m3_to_bf16_triton(fp8, e4m3_to_bf16_tensor)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles)

    gbps = lambda ms: size * fp8.element_size() * 2 * 1e-9 / (ms * 1e-3)  # Assuming read and write
    return gbps(ms), gbps(max_ms), gbps(min_ms)
