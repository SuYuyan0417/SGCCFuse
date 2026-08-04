import torch
import os as _os

device_ids = [0]
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 420
weight_decay = 1e-5
init_scale = 0.01
lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device = torch.device(f"cuda:{str(device_ids[0])}")
l_alpha = 3
l_beta = 1
l_gamma = 1
l_ks = 2
mse_w = 1
batch_size = 4
cropsize = 128
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5
cropsize_val = 256
batchsize_val = 1
shuffle_val = False
val_freq = 10
data_root = "./"
TRAIN_PATH = data_root + "data/train"
VAL_PATH = data_root + "data/test"
format_train = 'jpg'
format_val = 'jpg'
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False
MODEL_PATH = data_root + 'model/'
checkpoint_on_error = True
SAVE_freq = 50
suffix = 'model_checkpoint_00200.pt'
tain_next = False
trained_epoch = 0

_KNOWN_DATASETS = ['ir-vi']


def _auto_detect_dataset():
    data_dir = 'data'
    if not _os.path.exists(data_dir):
        return None, None
    existing = _os.listdir(data_dir)
    for ds in _KNOWN_DATASETS:
        if ds in existing and _os.path.isdir(_os.path.join(data_dir, ds)):
            return ds, ds
    return None, None


DATASET, DATASET_DIR = _auto_detect_dataset()

if DATASET is None:
    raise FileNotFoundError(
        f"未在 ./data/ 下找到已知数据集。\n"
        f"已知模式: {_KNOWN_DATASETS}\n"
        f"请将数据集文件夹放入 ./data/ (如 ./data/ir-vi/)"
    )
