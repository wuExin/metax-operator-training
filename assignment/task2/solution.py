import tilelang
import tilelang.language as T

@tilelang.jit
def tl_add_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    # Step 1: 用 T.ceildiv 计算 block 数。
    # Step 2: 在 T.Kernel 中取得 pid 并计算 base_idx。
    # Step 3: 用 T.Parallel 遍历 tile 元素。
    # Step 4: 判断 index < N，保护尾块越界。
    # Step 5: 写回 C[index] = A[index] + B[index]。
    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=128) as bx:
        base_idx = bx * BLOCK_N
        for i in T.Parallel(BLOCK_N):
            index = base_idx + i
            if index < N:
                C[index] = A[index] + B[index]
    return C
