import torch
import triton
import triton.language as tl

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

    t_sign = tl.where(t >= 0, 1, -1).to(tl.int16)
    t_abs = tl.abs(t)
    t_int16 = tl.view(t, tl.int16)

    uint16_t = tl.where(t_sign < 0, t_int16 + 32768, t_int16)

    # bf_bound_ptr 和 fp_bound_ptr  [bf_size * 2]  [fp_size * 2]
    # lower = bf_bound[uint16_t, 0]
    bf_lower = tl.load(bf_bound_ptr + uint16_t * 2, mask=mask, other=0.0)
    # dist = bf_bound[uint16_t, 1]
    bf_dist = tl.load(bf_bound_ptr + uint16_t * 2 + 1, mask=mask, other=0.0)
    # lower_v = fp_bound[uint16_t, 0]
    fp_lower_v = tl.load(fp_bound_ptr + uint16_t * 2, mask=mask, other=0.0)
    # upper_v = fp_bound[uint16_t, 1]
    fp_upper_v = tl.load(fp_bound_ptr + uint16_t * 2 + 1, mask=mask, other=0.0)

    rand = tl.rand(seed, offsets)

    fractional_part = (t_abs - bf_lower) / bf_dist
    rounded = tl.where(fractional_part >= rand, fp_upper_v, fp_lower_v)

    final_result = tl.where(t_sign < 0, rounded + 128, rounded)

    tl.store(out_ptr + offsets, final_result, mask=mask)

def sround_to_fp8_triton(
        t: torch.Tensor,
        bf_bound: torch.Tensor,
        fp_bound: torch.Tensor,
        out: torch.Tensor = None,
        seed: int = 12345
) -> torch.Tensor:
    assert bf_bound.dim() == 2 and bf_bound.size(1) == 2, "bf_bound 应该是 [N, 2] 的张量"
    assert fp_bound.dim() == 2 and fp_bound.size(1) == 2, "fp_bound 应该是 [N, 2] 的张量"
    assert t.device.type == 'cuda', "输入张量必须在 CUDA 设备上"

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
        BLOCK_SIZE=1024,
    )

    return out

if __name__ == "__main__":
    N = 10
    t = torch.randn(N, device='cuda', dtype=torch.bfloat16)

    bf_bound = torch.load("data/e5m2_bf_bound.pt", weights_only=True).to('cuda')
    fp_bound = torch.load("data/e5m2_fp8_bound.pt", weights_only=True).to('cuda')

    fp8_tensor = sround_to_fp8_triton(t, bf_bound, fp_bound)

    print(fp8_tensor)
