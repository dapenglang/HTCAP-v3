# HTCAP‑RS Demo (Single‑Step Heterogeneous Adversarial Purification)
See main.py and modules; no training required. Use FakeData to illustrate the pipeline.

Python 文件，`main.py` 统一调度。全部代码和文档都在这：

- 下载/查看仓库根目录
  - README.md
  - main.py
  - dct_utils.py
  - prox_tv.py
  - gating.py
  - hetero.py
  - purifier.py
  - attacks.py
  - smoothing.py
  - eval_pipeline.py

简要说明（和算法一一对齐）

- `dct_utils.py`：二维 DCT/IDCT（可微）、低/中/高频子带掩码。对应“异质子域分解（DCT 子带）”。
- `prox_tv.py`：**TV 一步近端**（Chambolle 风格单步近似）。对应“子域定制化抑制”。
- `gating.py`：**Sparsemax** 投影与像素级门控网络，输出 3 通道权重。对应“稀疏门控融合”。
- `hetero.py`：把 DCT→子带→近端→稀疏融合串成 **HeteroProjection**。
- `purifier.py`：占位净化器（Identity），便于你后续替换更强 backbone。
- `attacks.py`：纯 PyTorch **PGD-L∞/L2**（避免外部依赖，torchattacks 不必装）。
- `smoothing.py`：玩具版随机平滑证书估计（采样 pA/pB 与半径）。
- `eval_pipeline.py`：`FakeData` 数据集（无需下载），把净化+线性头组装成一个可评估模型。
- `main.py`：命令行入口，跑**干净推理/PGD 攻击/随机平滑估计**的全流程。

如何快速运行



'''

# 环境
conda create -n htcaprs-demo python=3.10 -y
conda activate htcaprs-demo
pip install torch torchvision numpy pyyaml autoattack

# 下载数据（CIFAR-10），干净评估 + PGD-L∞
python main.py --dataset cifar10 --download --attack none --batch-size 256 --device cuda
python main.py --dataset cifar10 --download --attack pgd --eps 8/255 --alpha 2/255 --steps 10 --norm linf --batch-size 128 --device cuda

# AutoAttack 标准配置（L∞, eps=8/255），示例只跑一个批次演示流程，可放大
python main.py --dataset cifar10 --download --autoattack --aa-eps 8/255 --aa-norm Linf --aa-version standard --aa-batch 128 --device cuda

# CIFAR-100 类似
python main.py --dataset cifar100 --download --attack none --device cuda
python main.py --dataset cifar100 --download --autoattack --aa-eps 8/255 --aa-norm Linf --aa-version standard --aa-batch 128 --device cuda

'''

设计取舍与注释风格

- 为保证**无下载/无训练**也能跑通，我用 `torchvision.datasets.FakeData` 生成 3×H×W 的样本，线性头随机初始化，目的是**演示管线**而不是给可比精度。
- DCT/IDCT 采用**矩阵分解实现**，梯度可回传；低频子带套一阶 **TV 近端**，中/高频默认恒等（你可以在 `hetero.py` 中换成带通卷积或轻量去噪器）。
- 稀疏门控采用 **Sparsemax**（在 `gating.py`），逐像素输出三路权重并做凸组合，避免“平均成糊”。
- 攻击是纯 PyTorch 版 **PGD**，避免外依赖；如果想接入 `autoattack`/`torchattacks`，我也能给你切换接口。
- `smoothing.py` 提供**可运行的玩具证书估计**，展示如何采样 pA/pB 并计算保守半径；实际论文版可替换为大样本高置信度估计。

想进一步升级吗？我可以：

- 把 `purifier.py` 换成轻量 ConvNeXt/UNet，并提供一份**极简微调脚本**；
- 在 `hetero.py` 加可选的**空间分块**（4×4）与 **BPDA 评测**；
- 接上真实数据集（CIFAR-10/100）和 **AutoAttack**，形成论文级基准脚本。





## V2

安排上了，直连 **CIFAR-10/100 + AutoAttack** 的“论文级基准”脚本已经写好并可下载运行：

- 仓库根目录： htcap_rs_demo
  - README.md
  - main.py
  - datasets.py（CIFAR-10/100 DataLoader）
  - autoattack_runner.py（AutoAttack 封装）
  - eval_pipeline.py（接入真实数据集）
  - 其余算法模块：`hetero.py`、`dct_utils.py`、`prox_tv.py`、`gating.py`、`purifier.py`、`attacks.py`、`smoothing.py`

快速使用

'''

# 环境
conda create -n htcaprs-demo python=3.10 -y
conda activate htcaprs-demo
pip install torch torchvision numpy pyyaml

# 1) 干净推理（无攻击）
python main.py --attack none

# 2) PGD-L∞ 演示
python main.py --attack pgd --eps 8/255 --alpha 2/255 --steps 10 --norm linf

# 3) PGD-L2 演示
python main.py --attack pgd --eps 1.0 --alpha 0.25 --steps 10 --norm l2

# 4) 随机平滑（玩具）半径估计
python main.py --attack none --rs-samples 1000 --rs-sigma 0.15

'''

要点说明（更贴近论文复现实操）

- **数据集**：`datasets.py` 提供 `cifar10_loaders/cifar100_loaders`，默认采用 CIFAR 均值方差归一化。`--download` 首次会自动拉取数据到 `--data-root`。
- **AutoAttack**：`autoattack_runner.py` 使用 `run_standard_evaluation` 封装，默认版本 `standard`，包含 APGD-CE、APGD-T、FAB-T、Square 等子攻击组合。
- **评估协议**：`main.py` 中 PGD 与 AA 分开配置；为演示速度，AA 默认只跑一个批次（`--aa-batch`），实际论文应对测试集分批全量评估。
- **算法模块**：
  - `hetero.py` 将 DCT 子带→TV 一步近端→Sparsemax 门控→融合串成 `HeteroProjection`；
  - `purifier.py` 仍为占位 `IdentityPurifier`（确保零训练也能跑通）；你可替换为轻量 CNN 或预训练 backbone；
  - `attacks.py` 的 PGD 实现纯 PyTorch，避免外依赖；
  - `smoothing.py` 给出玩具版随机平滑半径估计模板，便于扩展到论文版采样流程。

两点实用升级位（随时可加）

1. 将 `purifier.py` 切换为轻量 ConvNet，并提供 5~20 epoch 的快速微调脚本，以获得更有意义的鲁棒数值。
2. AutoAttack 批量评估脚本：对 `test_loader` 全量遍历并汇总 `aa_acc`，同时保存对抗样本或攻击日志，方便论文复核。

如果你希望强化与论文一致性，我可以继续补充：**统一随机种子、记录日志（wandb/CSV）、保存 AA 明细与可视化、PGD/AA 参数表**，再附一键 `bash` 脚本跑全套基准。
