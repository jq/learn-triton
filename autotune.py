import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config(
            BLOCK_SIZE=512,
            num_warps=4,
            num_stages=2
        ),
        triton.Config(
            BLOCK_SIZE=512,
            num_warps=8,
            num_stages=2
        ),
        triton.Config(
            BLOCK_SIZE=512,
            num_warps=16,
            num_stages=2
        ),
        triton.Config(
            BLOCK_SIZE=1024,
            num_warps=8,
            num_stages=2
        ),
        # Add more configurations as needed
    ],
    key=['n_elements'],  # Group configurations by the size of the input tensor
    num_repeats=3,        # Number of times to benchmark each configuration
    device='cuda'         # Specify the device; adjust if necessary
)
@triton.jit
def e4m3_to_bf16_triton_kernel(
        fp8_ptr,           # Pointer to input fp8 tensor
        lookup_ptr,        # Pointer to lookup table
        out_ptr,           # Pointer to output bf16 tensor
        n_elements,        # Number of elements
        BLOCK_SIZE: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    fp8_indices = tl.load(fp8_ptr + offsets, mask=mask, other=0).to(tl.int32)
    # In Triton, dynamic indexing on registers is not directly possible, so we load from HBM
    bf16 = tl.load(lookup_ptr + fp8_indices, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, bf16, mask=mask)

def e4m3_to_bf16_triton(
        fp8_tensor: torch.Tensor,
        lookup_table: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    assert fp8_tensor.dtype == torch.uint8, "Input tensor must be of type torch.uint8"
    assert lookup_table.dtype == torch.float16 or lookup_table.dtype == torch.bfloat16, \
        "Lookup table must be of type torch.float16 or torch.bfloat16"

    if out is not None:
        assert out.dtype == torch.bfloat16, "Output tensor must be of type torch.bfloat16"
        assert out.shape == fp8_tensor.shape, "Output tensor must have the same shape as input tensor"
    else:
        out = torch.empty_like(fp8_tensor, dtype=torch.bfloat16, device=fp8_tensor.device)

    n_elements = fp8_tensor.numel()

    # Calculate the grid size based on the BLOCK_SIZE from the autotuned kernel
    # Triton will handle selecting the best BLOCK_SIZE
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    # Launch the autotuned kernel
    e4m3_to_bf16_triton_kernel[grid](
        fp8_tensor,
        lookup_table,
        out,
        n_elements,
        BLOCK_SIZE=tl.constexpr,
        num_warps=tl.constexpr,
        num_stages=tl.constexpr
    )

    return out

def e4m3_to_bf16_torch(
        fp8_tensor: torch.Tensor, lookup_table: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:

    assert fp8_tensor.dtype == torch.uint8
    if out is not None:
        assert out.dtype == torch.bfloat16
        assert out.shape == fp8_tensor.shape
        return out.copy_(lookup_table[fp8_tensor.int()])
    return lookup_table[fp8_tensor.int()]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='e4m3_to_bf16-performance',
        args={},
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

    gbps = lambda ms: size * fp8.element_size() * 2 * 1e-9 / (ms * 1e-3)  # 假设读写
    return gbps(ms), gbps(max_ms), gbps(min_ms)
