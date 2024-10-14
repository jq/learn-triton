from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def fused_optimized_kernel(
        t_ptr, bf_bound_ptr, fp_bound_ptr, out_ptr, n_elements,
        seed, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    t = tl.load(t_ptr + offsets, mask=mask, other=0.0)

    t_abs = tl.abs(t)
    t_int16 = tl.cast(t, tl.int16, bitcast=True)
    pos_int16_t = t_int16 & 0x7FFF

    #torch.where(t_sign < 0, t_int16.int() + 32768, t_int16.int())
    # bf_bound_ptr 和 fp_bound_ptr  [bf_size * 2]  [fp_size * 2]
    # lower = bf_bound[uint16_t, 0]
    bf_lower = tl.load(bf_bound_ptr + pos_int16_t * 2, mask=mask, other=0.0)
    # dist = bf_bound[uint16_t, 1]
    bf_dist = tl.load(bf_bound_ptr + pos_int16_t * 2 + 1, mask=mask, other=0.0)
    # lower_v = fp_bound[uint16_t, 0]
    fp_lower_v = tl.load(fp_bound_ptr + pos_int16_t * 2, mask=mask, other=0.0)
    # upper_v = fp_bound[uint16_t, 1]
    fp_upper_v = tl.load(fp_bound_ptr + pos_int16_t * 2 + 1, mask=mask, other=0.0)


    rand = tl.rand(seed, offsets)

    fractional_part = (t_abs - bf_lower) / bf_dist
    rounded = tl.where(fractional_part >= rand, fp_upper_v, fp_lower_v)

    # final_result = (t_int16 & 0x8000) + rounded
    t_sign = (t_int16 >> 15) & 1
    # final_result = tl.where(t_sign == 1, rounded + 128, rounded)
    final_result = (t_sign << 7) | rounded
    tl.store(out_ptr + offsets, final_result, mask=mask)

def sround_to_fp8_triton(
        t: torch.Tensor,
        bf_bound: torch.Tensor,
        fp_bound: torch.Tensor,
        out: torch.Tensor = None,
        seed: int = 12345
) -> torch.Tensor:
    assert bf_bound.dim() == 2 and bf_bound.size(1) == 2
    assert fp_bound.dim() == 2 and fp_bound.size(1) == 2
    assert t.device.type == 'cuda'

    n_elements = t.numel()
    if out is None:
        out = torch.empty(n_elements, dtype=torch.uint8, device='cuda')

    bf_bound_flat = bf_bound.view(-1).contiguous()
    fp_bound_flat = fp_bound.view(-1).contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    fused_optimized_kernel[grid](
        t,
        bf_bound_flat,
        fp_bound_flat,
        out,
        n_elements,
        seed,
        BLOCK_SIZE=2048,
    )

    return out

def sround_to_fp8(t: Tensor, bf_bound: Tensor, fp_bound: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    randt = torch.rand(t.shape,  device='cuda', dtype=torch.float32)
    t_int16 = t.view(torch.int16)
    t_sign = torch.sign(t_int16)
    uint16_t = torch.where(t_sign < 0, t_int16.int() + 32768, t_int16.int())
    lower = bf_bound[uint16_t, 0]
    dist = bf_bound[uint16_t, 1]
    lower_v = fp_bound[uint16_t, 0]
    upper_v = fp_bound[uint16_t, 1]
    t = torch.abs(t)
    t = stochastic_rounding(t, randt, lower, dist, lower_v, upper_v, out)
    # can't be t.view(torch.uint8) because neg also map to positive
    # and we need t_sign to recover it
    return int8_to_uint8(t, t_sign, out)

def int16_to_uint16(t_int16: Tensor, t_sign: Tensor) -> Tensor:
    return torch.where(t_sign < 0, t_int16.int() + 32768, t_int16.int())

def stochastic_rounding(
        t: Tensor, randt: Tensor, lower: Tensor, dist: Tensor,
        lower_v: Tensor, upper_v: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    fractional_part = (t - lower) / dist
    if out is None:
        return torch.where(fractional_part>= randt, upper_v, lower_v)
    return torch.where(fractional_part>= randt, upper_v, lower_v, out=out)

def int8_to_uint8(t: Tensor, t_sign: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    if out is None:
        return torch.where(t_sign < 0, t + 128, t)
    return torch.where(t_sign < 0, t + 128, t, out=out)

def compare():
    # N = 10
    # t = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    # print(t)
    # tensor([-0.7109,  0.3672,  0.1953, -0.2051,  0.6250,  1.1562, -1.3750,  0.9805,
    #         -0.1104, -0.7812], device='cuda:0', dtype=torch.bfloat16)
    t = torch.tensor([-0.7109], device='cuda', dtype=torch.bfloat16) # 186
    bf_bound = torch.load("data/e5m2_bf_bound.pt", weights_only=True).to('cuda')
    fp_bound = torch.load("data/e5m2_fp8_bound.pt", weights_only=True).to('cuda')

    fp8_tensor = sround_to_fp8_triton(t, bf_bound, fp_bound)
    print(fp8_tensor)

    fp8_torch = sround_to_fp8(t, bf_bound, fp_bound)
    print(fp8_torch)
    # may not be the same since torch.rand and triton.rand are different
    #assert torch.allclose(fp8_tensor, fp8_torch, atol=1e-3), "Outputs do not match!"

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
    t = torch.randn(size, device='cuda', dtype=torch.bfloat16)

    bf_bound = torch.load("data/e5m2_bf_bound.pt", weights_only=True).to('cuda')
    fp_bound = torch.load("data/e5m2_fp8_bound.pt", weights_only=True).to('cuda')
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        func = lambda: sround_to_fp8(t, bf_bound, fp_bound)
    elif provider == 'triton':
        func = lambda: sround_to_fp8_triton(t, bf_bound, fp_bound)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles)

    gbps = lambda ms: size * t.element_size() * 2 * 1e-9 / (ms * 1e-3)  # Assuming read and write
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    # compare()
    benchmark.run(print_data=True, show_plots=True)

