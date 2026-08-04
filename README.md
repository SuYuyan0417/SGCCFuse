# SGCCFuse

**Saliency-Guided Cycle-Consistent Cross-Attention Fuse** — 显著性引导的循环一致交叉注意力多尺度红外-可见光图像融合网络。

基于双分支多尺度编码器（共享权重）+ 三尺度并行分支（SFE 显著性提取 + CPCA 循环一致交叉注意力）+ 由深到浅的轻量解码器，实现红外（IR）与可见光（VI）图像的高质量融合。

## 环境依赖

- Python 3.8+
- PyTorch 1.12+ （支持 `weights_only` 加载）
- torchvision, numpy, Pillow, tqdm, natsort

安装依赖：

```bash
pip install torch torchvision numpy pillow tqdm natsort
```

## 项目结构

```
.
├── sgccfuse.py            # 主模型 SGCCFuse
├── model.py               # Model 封装（加载权重、推理入口）
├── modules/
│   ├── sfe_block.py       # 创新点1：SFE 显著性特征提取
│   ├── cpca_block.py      # 创新点2：CPCA 循环一致渐进交叉注意力
│   ├── shared_encoder.py  # 共享编码器
│   ├── scale_block.py     # ScaleBlock（ConvNeXt-v2 风格基础块）
│   └── module_util.py     # 工具函数（initialize_weights）
├── train_logging.py       # 训练脚本
├── test.py                # 测试/推理脚本
├── datasets_MSRS.py       # 数据集加载（train/val）
├── config.py              # 配置（自动检测数据集）
├── data/                  # 数据集目录（需自行放入）
├── model/                 # 预训练权重
└── result/                # 测试结果输出目录
```

## 数据准备

### 测试集

`test.py` 会自动检测 `./data/` 下的数据集文件夹（支持 `ir-vi` / `MRI-CT` / `MRI-PET` / `MRI-SPECT`）。
以红外-可见光融合为例，将测试集按如下结构放入 `./data/ir-vi/`：

```
data/ir-vi/
├── ir/        # 红外图像（RGB 或转 RGB）
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── vi/        # 可见光图像（与 ir 同名一一对应）
    ├── 001.png
    ├── 002.png
    └── ...
```

> 注意：`ir/` 与 `vi/` 下文件名需一一对应，脚本按文件名配对。

### 训练集

`datasets_MSRS.py` 规定训练数据放在固定路径：

```
data/train/ir/    data/train/vi/    # 训练：红外 / 可见光
```

## 测试（推理）

直接运行：

```bash
python test.py
```

- 输出的融合图像保存在 `./result/ir-vi/`（目录名与数据集名一致）。
- 默认使用预训练权重 `./model/a3_b1_c1_d2/20260726_1625_1/model_checkpoint_00400.pt`。

## 训练

训练代码将在论文正式接受后发布。

## 核心创新点

1. **SFE（Saliency Feature Extraction，显著性特征提取）**：在每一尺度对单模态特征做通道+空间双注意力显著性加权，突出关键目标区域。
2. **CPCA（Cycle-Consistent Progressive Cross-Attention，循环一致渐进交叉注意力）**：通过循环一致掩码（`j[i]==i` 闭环）逐步强化可靠的跨模态对应，双向（IR↔VI）对称交互，并以可学习门控残差渐进注入跨模态信息。


