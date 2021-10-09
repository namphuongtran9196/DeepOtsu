from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import _reduction as _Reduction
from torch.nn import functional as F


class HeScho(nn.Module):
    def __init__(self):
        super(HeScho, self).__init__()

    def forward(self, outputs, gt):
        losses = [torch.mean(torch.abs(gt - output)) for output in outputs]
        loss = torch.mean(torch.stack(losses))
        return loss


class _Loss(nn.Module):
    reduction: str

    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(
                size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]


class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce,
                                               reduction)
        self.ignore_index = ignore_index

    def forward(self, inputs: List, target: Tensor) -> Tensor:
        # inputs: list of tensors of shape (B, C, H, W) of length num_block
        #         [(B, C, H, W), (B, C, H, W)]
        # target: tensor of shape (B, C, H, W)
        num_block = len(inputs)
        batch_size = inputs[0].shape[0]
        losses = 0.0
        for i in range(batch_size):
            for j in range(num_block):
                losses += F.cross_entropy(inputs[j][i],
                                          torch.tensor(target[i],
                                                       dtype=torch.long),
                                          weight=self.weight,
                                          ignore_index=self.ignore_index,
                                          reduction=self.reduction)
        return losses / (num_block * batch_size)
