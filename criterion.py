import torch
from torch import nn


class HeScho(nn.Module):
    def __init__(self):
        super(HeScho, self).__init__()

    def forward(self, output, gt):
        loss = torch.mean(torch.abs(gt - output))
        return loss
