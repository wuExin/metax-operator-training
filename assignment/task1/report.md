# 任务一环境报告：沐曦 GPU 与 MXMACA 环境

- 日期：2026-09-14
- 平台：模力方舟（算力券租用）
- 实例：主机 ID `jiajia-mxc500-0004`，实例 ID `JTDJUMAEXKTMT1M`
- 规格：曦云 C500，sGPU 切分 16 GB × 1，按量计费
- 镜像：TileLang / 0.1.9 / Python 3.12 / maca 3.3.0.4（实际环境 MACA 版本更新，见下）

## 1. 硬件识别（mx-smi）

命令：

```bash
mx-smi
```

关键输出（截图见 `mx-smi.png`）：

```
mx-smi version: 2.2.12
Timestamp: Mon Sep 14 09:42:47 2026
Attached GPUs: 1
MX-SMI 2.2.12                 Kernel Mode Driver Version: 3.8.30
MACA Version: 3.5.3.20        BIOS Version: 1.33.5.0

Board            GPU  Persist-M  Bus-id        GPU-Util    sGPU-M
Pwr:Usage/Cap    Temp Perf       Memory-Usage  GPU-State
0  MetaX C500    0    Off        0000:0f:00.0  0%          Enabled
48W / 350W       37C  P0         1058/65536 MiB  Available

Sliced GPU
Minor  GPU  sGPU-Id  Compute  Vram Quota    sGPU-Util
002    0    2        25%      0/16000 MiB   0%

Process: no process found
```

结论：系统成功识别 1 张 MetaX C500（物理卡显存 65536 MiB，即 64 GB），
当前通过 sGPU 切分出 16 GB 配额（16000 MiB）供本实例使用；卡温 37C、功耗 48W/350W、
状态 Available，环境可正常使用。

## 2. 软件栈识别

命令：

```bash
pip list | grep -iE "torch|maca|metax"
```

关键包（截图见 `pip-list.png`）：

| 包 | 版本 | 说明 |
|---|---|---|
| torch | 2.8.0+metax3.5.3.9 | 沐曦适配版 PyTorch |
| torchaudio | 2.4.1+metax3.5.3.9 | |
| torchvision | 0.15.1+metax3.5.3.9 | |
| triton | 3.0.0+metax3.5.3.9 | 沐曦适配版 Triton |
| flash_attn | 2.6.3+metax3.5.3.9torch2.8 | 注意力加速库 |
| flashinfer | 0.2.6+metax3.5.3.9torch2.8 | |
| xformers | 0.0.22+metax3.5.3.9torch2.8 | |
| vllm_metax | 0.19.0+…maca3.5.3.20.torch2.8 | 沐曦适配版 vLLM |
| mcoplib | 0.4.4+maca3.5.3.20.torch2.8 | |
| deep_gemm / lmcache / nixl | maca3.5.3.5xx | |

结论：均为 `+metax…` / `+maca…` 后缀的沐曦适配版本，证明 Python 软件栈与
MXMACA 生态正确匹配。

## 3. 补充系统信息

```bash
uname -a          # Linux 0ec5830a4b35 5.15.0-58-generic x86_64 GNU/Linux
cat /etc/os-release | head -2   # Ubuntu 22.04.3 LTS
python --version  # Python 3.12.11
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.8.0+metax3.5.3.9 True
```

`torch.cuda.is_available()` 返回 True（MACA 版 PyTorch 复用 CUDA 接口），
PyTorch 可正常使用沐曦 GPU。

## 4. 环境小结

| 项目 | 值 |
|---|---|
| GPU | MetaX 曦云 C500 × 1（sGPU 16 GB 配额） |
| 驱动 / mx-smi | Kernel Driver 3.8.30 / mx-smi 2.2.12 |
| MACA | 3.5.3.20（mx-smi 报告值） |
| OS | Ubuntu 22.04.3 LTS（内核 5.15.0-58-generic） |
| Python | 3.12.11 |
| PyTorch | 2.8.0+metax3.5.3.9（cuda 可用） |
| TileLang | 0.1.9（镜像预装） |

环境就绪，可以进入任务二（TileLang Add 与 NineToothed Vector Add）。

## 附：验收截图

- `mx-smi.png`：mx-smi 完整输出
- `pip-list.png`：pip list 过滤 torch/maca/metax 输出
