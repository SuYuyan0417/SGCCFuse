import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted

TRAIN_PATH = c.TRAIN_PATH
VAL_PATH = c.VAL_PATH
format_train = c.format_train
format_val = c.format_val


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files1 = natsorted(sorted(glob.glob(TRAIN_PATH + "/ir/*")))
            self.files2 = natsorted(sorted(glob.glob(TRAIN_PATH + "/vi/*")))
        else:
            self.files1 = sorted(glob.glob(VAL_PATH + "/ir/*"))
            self.files2 = sorted(glob.glob(VAL_PATH + "/vi/*"))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files1[index])
            image = to_rgb(image)
            ir = T.ToTensor()(image)
            image1 = Image.open(self.files2[index])
            image1 = to_rgb(image1)
            vi = T.ToTensor()(image1)
            cat = torch.cat([ir, vi], dim=0)
            cat = self.transform(cat)
            return cat[:3], cat[3:]
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files1)


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
])

trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)

testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)
