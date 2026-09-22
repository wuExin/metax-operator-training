# 任务二报告：TileLang Add 与 NineToothed Vector Add

- 日期：2026-09-22
- 平台：模力方舟（算力券租用），沐曦 MetaX C500
- 软件栈：TileLang 0.1.9（`/app/tilelang-metax`）+ PyTorch 2.8.0+metax3.5.3.9 + MACA 3.5.3.20
- 环境：Python 3.12.11（`/opt/conda`），运行前 `source ./setup_env.sh` 加载 TileLang 与 MACA 环境变量
- 代码：`assignment/task2/solution.py`（TileLang）、`assignment/task2/ninetoothed_add.py`（NineToothed）

## 1. 任务理解

实现一维 `float16` 向量加法 `C[i] = A[i] + B[i]`，核心考验三点：

1. **分块并行**：把长度为 N 的向量切成每组 `BLOCK_N` 个元素，交给多个 block 同时计算；
2. **尾块保护**：N 不整除 BLOCK_N 时（如 N=127、100003），最后一个 block 必须用
   `if index < N` 挡住越界读写；
3. **正确性校验与性能对比**：与 PyTorch 逐元素比对，并对比至少两种 BLOCK_N 的耗时。

分别用 TileLang（手写 kernel：自己管 block、索引、边界）和 NineToothed（声明式：
只描述切分方式和 tile 内计算，其余由框架生成）两种范式各实现一遍。

## 2. A 部分：TileLang Vector Add

### 2.1 实现要点（`solution.py`）

```python
@tilelang.jit
def tl_add_1d(A, B, BLOCK_N: int):
    N = T.const("N")
    ...
    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=128) as bx:   # block 数 = 向上取整
        base_idx = bx * BLOCK_N                                # 本 block 负责的起点
        for i in T.Parallel(BLOCK_N):                          # tile 内并行
            index = base_idx + i
            if index < N:                                      # 尾块保护
                C[index] = A[index] + B[index]
    return C
```

- `T.const("N")` 使 N 成为符号变量，同一份编译产物可服务不同长度；
- `T.ceildiv(N, BLOCK_N)` 计算 block 数，保证覆盖全部元素；
- `T.Parallel` 展开为 tile 内并行访存，无需手写线程索引。

### 2.2 正确性（`pytest test_add.py`）

N = 1、127、1024、100003、1048576 五个规模全部通过：

```
5 passed in 11.19s
```

`max_abs_error` 均为 `0.000e+00`，fp16 下与 PyTorch 逐元素完全一致。

### 2.3 性能（`benchmark_add.py`，两种 BLOCK_N）

BLOCK_N = 1024（耗时单位 us）：

| N | 正确 | TileLang | PyTorch |
|---:|:---:|---:|---:|
| 1 | PASS | 17.28 | 11.80 |
| 127 | PASS | 16.18 | 11.58 |
| 1024 | PASS | 16.68 | 11.51 |
| 100003 | PASS | 16.82 | 11.43 |
| 1048576 | PASS | 17.01 | 12.28 |

BLOCK_N = 256：

| N | 正确 | TileLang | PyTorch |
|---:|:---:|---:|---:|
| 1 | PASS | 14.23 | 8.00 |
| 127 | PASS | 12.94 | 7.94 |
| 1024 | PASS | 13.66 | 8.23 |
| 100003 | PASS | 13.17 | 8.44 |
| 1048576 | PASS | 18.20 | 11.77 |

### 2.4 结果分析

- **耗时曲线平坦**：从 N=1 到 N=1048576（数据量 2 MB），TileLang 耗时基本不变
  （约 13~18 us），说明小规模下瓶颈是 kernel 启动等固定开销，远未触及 C500 的
  显存带宽；
- **BLOCK_N 对比**：两种块大小整体差距不大；最大规模 2^20 时 BLOCK_N=1024 略占优
  （17.01 vs 18.20 us），符合"较大块减少调度开销、尾部浪费占比也更小"的预期；
  小规模下两者都在固定开销区间内，差异属于运行间噪声；
- **TileLang 慢于 PyTorch** 属正常：逐元素加法是访存受限算子，PyTorch 原生
  kernel 已深度优化，任务说明也明确不要求 TileLang 更快。

## 3. B 部分：NineToothed Vector Add

### 3.1 实现要点（`ninetoothed_add.py`）

```python
def arrangement(lhs, rhs, output):
    return (
        lhs.tile((BLOCK_SIZE,)),
        rhs.tile((BLOCK_SIZE,)),
        output.tile((BLOCK_SIZE,)),
    )

def application(lhs, rhs, output):
    output = lhs + rhs
```

`arrangement()` 只声明"按 BLOCK_SIZE=1024 切块"，尾块掩码、启动配置全部由框架
自动处理；`application()` 只写数学表达。默认 `BLOCK_SIZE=1024`，
`NINETOOTHED_AUTOTUNE=1` 时框架在 256~1024 间自动搜索。

### 3.2 正确性与性能

```
1 passed in 8.02s            # size=98432 与 PyTorch allclose 通过
```

| size | 正确 | NineToothed(ms) | PyTorch(ms) |
|---:|:---:|---:|---:|
| 2^18 | PASS | 0.0320 | 0.0312 |
| 2^20 | PASS | 0.0361 | 0.0340 |
| 2^22 | PASS | 0.0547 | 0.0451 |
| 2^24 | PASS | 0.0792 | 0.0677 |
| 2^26 | PASS | 0.2189 | 0.1657 |
| 2^27 | PASS | 0.7656 | 0.5582 |

（完整 10 个规模 2^18~2^27 全部 PASS，此处节选）

NineToothed 比 PyTorch 慢约 3%~37%，规模越大差距越明显：访存受限算子上
PyTorch 原生实现已接近带宽极限，生成代码的索引计算与调度开销在大规模下占比升高。

### 3.3 开发体验记录（NineToothed 分块方式与对比 TileLang）

NineToothed 采用声明式开发：`arrangement()` 只描述把 1-D 张量按 `BLOCK_SIZE`
切成 tile，`application()` 只写 `output = lhs + rhs` 一句数学表达，尾块保护、
越界掩码、kernel 启动和线程索引全部由框架自动生成。对比 TileLang 需要手写
`T.Kernel` 启动、`T.Parallel` 循环和 `if index < N` 边界判断，NineToothed
代码量更小、心智负担更低，把"怎么算在哪个 tile 上"和"算什么"分开了。
实际踩的坑：PyPI 最新版 `Tensor.tile()` 要求传入可迭代的 shape
（如 `(BLOCK_SIZE,)`），直接传整数会报 `TypeError`，与官方文档示例略有出入。
性能上两种范式的简单逐元素算子都到了固定开销/带宽瓶颈区间，差距主要来自
生成代码质量而非开发范式本身。

## 附：验收截图清单

| 截图 | 内容 |
|---|---|
| `test_add.png` | TileLang `pytest`：5 passed |
| `benchmark_add_1024.png` | TileLang benchmark，BLOCK_N=1024 |
| `benchmark_add_256.png` | TileLang benchmark，BLOCK_N=256 |
| `test_ninetoothed.png` | NineToothed `pytest`：1 passed |
| `benchmark_ninetoothed.png` | NineToothed benchmark（2^18~2^27） |
