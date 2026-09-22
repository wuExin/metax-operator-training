---
name: metax-ssh
description: 'SSH 远程操作模力方舟（沐曦 MetaX C500 / MXMACA）GPU 云实例。Use when user asks to 操作模力方舟/沐曦实例、远程跑 TileLang/MACA 测试或 benchmark、上传文件到 GPU 云机、查看 mx-smi 等。封装了连接认证、引号网关绕过、conda/MACA 环境加载、SFTP 上传的完整工作流与踩坑对策。'
---

# 模力方舟沐曦 GPU 实例远程操作

## Overview

通过 paramiko（Python SSH 库）从本地 Windows 远程操作模力方舟租用的沐曦 C500 实例：
执行命令、上传文件、跑 TileLang/MACA 测试与 benchmark。本 skill 固化了三类坑的解法：
SSH 网关认证、网关吞引号、云端 conda/MACA 环境。

## 第一步：向用户索取连接信息（每次触发必做）

实例是按量租用的，**连接信息每次都会变，不要假设、不要复用旧值、不要写入任何文件**。
先向用户询问以下四项（用户通常从模力方舟控制台复制）：

1. **SSH 地址**（IP 或域名）——注意控制台截图里此栏常被打码，必须让用户以文字提供
2. **端口**
3. **用户名**（平台格式为 `root+vm-<实例ID>`）
4. **密码**

用户提供截图时，先核对你需要的字段是否可见，缺哪项就明确问哪项。拿到后设置环境变量：

```powershell
$env:SSH_HOST="<地址>"; $env:SSH_PORT="<端口>"
$env:SSH_USER="<用户名>"; $env:SSH_PASS="<密码>"
$env:SSH_TIMEOUT="600"   # 长任务（JIT 编译/benchmark）给足超时
```

## 连接后：预期环境（不符时先排查再继续）

以下是基于标准镜像（TileLang / 0.1.9 / Python 3.12 / maca）的预期值，新实例上先验证：

- 云端仓库：`/data/gollamago`（缺失则从 gitlink 克隆，国内云机访问快）
- GPU：MetaX C500 64GB（`mx-smi` 确认）
- MACA：`/opt/maca`；TileLang 预装于 `/app/tilelang-metax`
- conda Python 3.12：`/opt/conda/bin/python`

## 工具脚本（本 skill 目录下 `scripts/`）

- `ssh_run.py`：远程执行命令。环境变量传 `SSH_HOST/SSH_PORT/SSH_USER/SSH_PASS/SSH_TIMEOUT`，
  命令作为 argv[1]，输出回显并带 `[exit=N]`。
- `ssh_put.py`：SFTP 上传。argv 依次为本地路径、远端路径。

两个脚本都必须用 `allow_agent=False, look_for_keys=False` 连接，否则网关认证会报
`AuthenticationException: transport shut down or saw EOF`。

## 关键坑与对策（必读）

### 1. 平台 SSH 网关会吞掉命令里的引号

含 `"` `'` 的命令经网关转发后引号会被剥离，导致远端 shell 拆词错误。
**对策：复杂命令一律 base64 编码后解码执行**，远端只出现字母数字：

```powershell
$b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\job.sh"))
python <skill目录>\scripts\ssh_run.py "echo $b64 | base64 -d | bash"
```

本地写脚本文件 → base64 → 远端 `| bash`，这是最可靠通道。简单无引号命令（如
`mx-smi`、`ls /data`）可直接传。

### 2. PowerShell 引号规则

- 远程命令用**单引号**包裹（verbatim，不做插值）
- 绝对不要用 `\"` 转义双引号——PowerShell 会把内容当本地命令执行

### 3. 云端非交互 shell 没有 python

`python` 不在 PATH，必须先：

```bash
export PATH=/opt/conda/bin:$PATH    # Python 3.12.11
```

### 4. TileLang / MACA 环境必须 source

triton（metax 版）import 时读取 MACA 环境变量，不 source 直接 import 会报
`TypeError: expected str ... not NoneType`（`maca_home_dirs()` 返回 None）。任何跑
GPU 代码的脚本开头必须是：

```bash
export PATH=/opt/conda/bin:$PATH
cd /data/gollamago
source ./setup_env.sh        # 设置 TILELANG_ROOT、MACA_PATH、LD_LIBRARY_PATH
```

注意 `source` 要在 bash 内完成——所以远程命令走 `| bash` 正好满足。

### 5. 云端装 Python 包

- 用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`
- 必须 `--no-deps`：防止 pip 把已装的 metax 定制版 triton/torch 换成 CUDA 版
  （例：`pip install --no-deps ninetoothed`）

### 6. 文件修改用 SFTP 上传

本地改好 → `ssh_put.py` 上传覆盖远端文件。不要在远端手写文件（网关吞引号，
heredoc 不可靠）。仓库缺失时先克隆：
`git clone https://gitlink.org.cn/ccf-ai-infra/gollamago.git /data/gollamago`

## 标准工作流

```powershell
# 前置：向用户索取连接信息并设置环境变量（见"第一步"）
# 每个新 shell 会话都要重新设置

# 1. 探环境
python ssh_run.py "mx-smi | head -15"

# 2. 写任务脚本（本地）
#    job.sh 开头必须包含：export PATH=/opt/conda/bin:$PATH + source ./setup_env.sh

# 3. base64 执行
$b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes("job.sh"))
python ssh_run.py "echo $b64 | base64 -d | bash 2>&1 | tail -20"

# 4. 上传本地改动的文件
python ssh_put.py <本地文件> /data/gollamago/<远端路径>
```

## 云端速查

| 用途 | 命令 |
|---|---|
| GPU 状态 | `mx-smi`（Task 1 验收材料） |
| TileLang 测试 | `cd /data/gollamago/assignment/task2 && python -m pytest -q test_add.py` |
| TileLang benchmark | `python benchmark_add.py`（改 BLOCK_N：`sed 's/BLOCK_N=1024/BLOCK_N=256/' benchmark_add.py > /tmp/b256.py && python /tmp/b256.py`） |
| NineToothed | `python -m pytest -q test_ninetoothed_add.py && python benchmark_ninetoothed_add.py` |
| TileLang 首次运行 | 含 JIT 编译，明显偏慢属正常，性能结论必须看预热后数据 |

## 经验教训

- TileLang 演示 kernel、NineToothed `tile()` API 等版本差异常见：云端行为和文档
  不一致时，先 `grep -n "def xxx" <site-packages路径>` 看安装版真实签名再改代码。
- 截图类交付物让用户自己在 Jupyter Terminal 重跑（命令可从脚本内容提取），
  数据以用户终端为准；Agent 跑通是为了先验证可行性。
