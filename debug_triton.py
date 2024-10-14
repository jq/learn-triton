'''https://github.com/Deep-Learning-Profiling-Tools/triton-viz
static_print and static_assert are intended for compile-time debugging.

device_print and device_assert are used for runtime debugging.
device_assert executes only when TRITON_DEBUG is set to 1
set the environment variable TRITON_INTERPRET to 1 using numpy equivalents of Triton operations.
For debugging on NVIDIA GPUs, compute-sanitizer is an effective tool for checking data races and memory access issues.
 To use it, prepend compute-sanitizer to your command to run the Triton program.

 num_warps：指定用于执行内核的 warp 数量。Warp 是 GPU 执行的基本单位，每个 warp 通常由 32 个线程组成。
num_stages：指定在软件流水线中预先加载的内存数据数量，以提高内存带宽利用率。
num_warps 决定了每个 Triton 块所包含的线程数。通常，num_warps 越大，GPU 中的并行度越高，
理论上可以提升内核的执行效率，但也会增加寄存器和共享内存的压力。
一个 warp 通常由 32 个线程组成，因此 num_warps=1 表示一个线程块有 32 个线程，而 num_warps=4 表示有 128 个线程（32×4）。
A100 具有 64K 32-bit 寄存器（即 65536 个寄存器）可供每个多处理器（Streaming Multiprocessor, SM）使用。
如果一个内核使用了过多的寄存器，那么 SM 上能同时执行的 warp 数量就会减少。num_warps 设置得过高时可能导致寄存器溢出。
每个 SM 在 A100 上可以支持最多 32 个线程块 和 64 个 warp。
因此，num_warps 的理论上限是 64。

使用 Triton 编写内核后，运行内核时使用 Nsight 进行分析，你可以找到瓶颈，例如：
GPU 是否受到内存延迟限制？
是否达到了寄存器或共享内存的上限？
nsys profile -o profile_report python your_script.py

在 A100 这种 GPU 上，性能瓶颈通常来自以下两个方面：
内存带宽瓶颈：需要增大 num_stages 来隐藏内存访问延迟。
计算瓶颈：需要增大 num_warps 来提高并行度。
num_warps：对于计算密集型内核，尝试设置为 4-8 个 warp，充分利用并行计算资源。
num_stages：对于内存访问频繁的内核，尝试设置为 2-4 个 stage，隐藏内存延迟。

'''
import os

# os.environ['TRITON_DEBUG'] = '1'
# os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
from triton.language import device_print
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=8),
    ],
    key=["N"]
)
@triton.jit
def kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    X_val = tl.load(X + offsets, mask=mask)
    Y_val = X_val * 2
    tl.store(Y + offsets, Y_val, mask=mask)

N = 1024
x = triton.testing.random((N,), device='cuda')
y = triton.testing.empty((N,), device='cuda')

# 自动选择最佳配置
kernel[(1,)](x, y, N)


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
    device_print(offsets)
    fp8_indices = tl.load(fp8_ptr + offsets, mask=mask, other=0).to(tl.int32)
    # 在 Triton 无法直接对加载到寄存器中的缓存进行动态索引，所以只能从HBM加载
    bf16 = tl.load(lookup_ptr + fp8_indices, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, bf16, mask=mask)