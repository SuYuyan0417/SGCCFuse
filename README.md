# SGCCFuse

**Saliency-Guided Cycle-Consistent Cross-Attention Fuse** — a saliency-guided, cycle-consistent cross-attention network for multi-scale infrared-visible image fusion.

Built upon a dual-branch multi-scale encoder (with weight sharing) plus three parallel branches at three scales (SFE saliency extraction + CPCA cycle-consistent cross-attention) and a lightweight decoder running from deep to shallow layers, the network achieves high-quality fusion of infrared (IR) and visible (VI) images.

## Environment Requirements

- Python 3.8+
- PyTorch 1.12+ (supports `weights_only` loading)
- torchvision, numpy, Pillow, tqdm, natsort

Install dependencies:

```bash
pip install torch torchvision numpy pillow tqdm natsort
```

## Project Structure

```
.
├── sgccfuse.py            # Main model SGCCFuse
├── model.py               # Model wrapper (weight loading, inference entry)
├── modules/
│   ├── sfe_block.py       # Innovation 1: SFE saliency feature extraction
│   ├── cpca_block.py      # Innovation 2: CPCA cycle-consistent progressive cross-attention
│   ├── shared_encoder.py  # Shared encoder
│   ├── scale_block.py      # ScaleBlock (ConvNeXt-v2 style basic block)
│   └── module_util.py      # Utility functions (initialize_weights)
├── train_logging.py       # Training script
├── test.py                # Test/inference script
├── datasets_MSRS.py       # Dataset loader (train/val)
├── config.py              # Configuration (auto-detect dataset)
├── data/                  # Dataset directory (to be populated by the user)
├── model/                 # Pretrained weights
└── result/                # Test output directory
```

## Data Preparation

### Test Set

`test.py` automatically detects dataset folders under `./data/` (supports `ir-vi`).
For infrared-visible fusion, place the test set under `./data/ir-vi/` with the following structure:

```
data/ir-vi/
├── ir/        # Infrared images (RGB or converted to RGB)
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── vi/        # Visible images (paired one-to-one with ir by file name)
    ├── 001.png
    ├── 002.png
    └── ...
```

> Note: File names under `ir/` and `vi/` must correspond one-to-one; the script pairs images by file name.

### Training Set

`datasets_MSRS.py` specifies that training data should be placed at fixed paths:

```
data/train/ir/    data/train/vi/    # Training: infrared / visible
```

## Testing (Inference)

Run directly:

```bash
python test.py
```

- Fused output images are saved to `./result/ir-vi/` (the directory name matches the dataset name).
- By default, `test.py` loads the pretrained weights from `./model/best.pt` (see [Pretrained Weights](#pretrained-weights) for how to obtain them).

## Training

The training code will be released after the paper is formally accepted.

## Pretrained Weights

The trained model weights (`best.pt`, 136.4 MB) are shared via Baidu Netdisk:

- **Download:** [best.pt (Baidu Netdisk)](https://pan.baidu.com/s/1M3o2ZhRI_-eOwGE_Ork2JA?pwd=0417)
- **Extraction code:** `0417`

After downloading, place the file at `./model/best.pt` (create the `model/` directory if it does not exist). The inference script `test.py` loads this path by default.

## Core Innovations

1. **SFE (Saliency Feature Extraction)**: At each scale, applies channel- and spatial-attention-based saliency weighting to single-modal features, highlighting key target regions.
2. **CPCA (Cycle-Consistent Progressive Cross-Attention)**: Gradually strengthens reliable cross-modal correspondences through a cycle-consistent mask (`j[i]==i` closed loop), performs bidirectional (IR↔VI) symmetric interaction, and progressively injects cross-modal information via learnable gated residuals.
