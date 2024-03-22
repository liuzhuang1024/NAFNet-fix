from basicsr.models import losses

import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class ConcatLoss(nn.Module):
    def __init__(self, name:list) -> None:
        super().__init__()
        self.fn = []
        for i in name:
            type = i.pop('type')
            self.fn.append(
                (type, getattr(losses, type)(**i))
            )

    def forward(self, pred, target, loss_dict:dict):
        total_loss = 0
        for key, fn in self.fn:
            loss = fn(pred, target)
            loss_dict.update({f"l_{key}": loss})
            total_loss += loss
        return total_loss
            
        