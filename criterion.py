import torch
from torch import nn


class HeScho(nn.Module):
    def __init__(self):
        super(HeScho, self).__init__()

    def forward(self, outputs, gt):
        losses = [torch.mean(torch.abs(gt - output)) for output in outputs]
        loss = torch.mean(torch.stack(losses))
        return loss
