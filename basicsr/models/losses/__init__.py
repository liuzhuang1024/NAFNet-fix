# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .losses import (L1Loss, MSELoss, PSNRLoss, MS_SSIMLoss, LPIPSLoss)
from .newloss import ConcatLoss
from .content_loss import PerceptualLoss, FFTloss

__all__ = [
    'FFTloss', 'L1Loss', 'MSELoss', 'PSNRLoss', 'MS_SSIMLoss', 'LPIPSLoss', 'ConcatLoss', 'PerceptualLoss'
]
