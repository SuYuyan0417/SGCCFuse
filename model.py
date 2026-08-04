import torch.optim
import torch.nn as nn
import config as c
from sgccfuse import SGCCFuse


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = SGCCFuse(in_channels=3, out_channels=3)

    def forward(self, x):
        return self.model(x)


def init_model(mod):

    pass
