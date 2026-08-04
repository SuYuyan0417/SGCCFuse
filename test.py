import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import os
import shutil
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import time
import torch.nn.functional as F
from tqdm import tqdm

cuda_my = f"cuda:{str(c.device_ids[0])}"
device = torch.device(cuda_my)


def load(net, name):
    try:
        state_dicts = torch.load(name, map_location=cuda_my, weights_only=True)
    except TypeError:
        state_dicts = torch.load(name, map_location=cuda_my)
    net.load_state_dict(state_dicts['net'])



def pad_to_multiple(x, multiple=8):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, (pad_h, pad_w)


def infer_one(net, data0, data1, dev):
    with torch.no_grad():
        cover = data0.to(dev)
        secret = data1.to(dev)
        orig_h, orig_w = cover.shape[-2], cover.shape[-1]
        cover, _ = pad_to_multiple(cover, multiple=8)
        secret, _ = pad_to_multiple(secret, multiple=8)
        input_img = torch.cat((cover, secret), 1)
        output = net(input_img)
        output = output[..., :orig_h, :orig_w]
    return output


def test(dataset_name, data_root, test_out_path, data_folder_name=None):
    if data_folder_name is None:
        data_folder_name = dataset_name
    Time = []
    test_folder = os.path.join(data_root, data_folder_name)
    test_out_folder = os.path.join(test_out_path, dataset_name)
    if os.path.exists(test_out_folder):
        shutil.rmtree(test_out_folder)
    os.makedirs(test_out_folder)
    ds_key = dataset_name.rstrip('-')
    if ds_key in ['ir-vi', 'IR-VIS']:
        model_path = './model/model-best.pt'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    net = Model()
    net.to(device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    load(net, model_path)
    net.eval()
    subfolders = sorted([
        folder_name for folder_name in os.listdir(test_folder)
        if os.path.isdir(os.path.join(test_folder, folder_name))
    ])
    if len(subfolders) != 2:
        raise ValueError(f"Expected exactly 2 subfolders in {test_folder}, got {subfolders}")
    folder_a, folder_b = subfolders[0], subfolders[1]
    img_names = sorted(os.listdir(os.path.join(test_folder, folder_a)))
    for img_name in tqdm(img_names, desc=f"测试 {dataset_name}", unit="张"):
        path_a = os.path.join(test_folder, folder_a, img_name)
        path_b = os.path.join(test_folder, folder_b, img_name)
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        crop_h = min(img_a.height, img_b.height)
        crop_w = min(img_a.width, img_b.width)
        center_crop = T.CenterCrop((crop_h, crop_w))
        img_a = center_crop(img_a)
        img_b = center_crop(img_b)
        data0 = img_a
        data1 = img_b
        data0 = Variable(ToTensor()(data0)).unsqueeze(0)
        data1 = Variable(ToTensor()(data1)).unsqueeze(0)
        tic = time.time()
        output = infer_one(net, data0, data1, device)
        end = time.time()
        Time.append(end - tic)
        torchvision.utils.save_image(output, os.path.join(test_out_folder, img_name))
        del output, data0, data1, img_a, img_b
        torch.cuda.empty_cache()
    Time = Time[2:len(Time) - 2]
    return (sum(Time))


if __name__ == '__main__':
    data_root = f'./data'
    test_out_folder = f'./result'
    dataset_name = c.DATASET
    data_folder_name = c.DATASET_DIR
    print(f"当前数据集: {dataset_name} (文件夹: {data_folder_name})")
    test_time_avg = test(dataset_name, data_root, test_out_folder, data_folder_name)
    print(test_time_avg)
