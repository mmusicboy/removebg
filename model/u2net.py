# Original U2NET model architecture
# Source: https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py

import torch
import torch.nn as nn

# Здесь мы вставим только заглушку, чтобы не перегружать рендер.
# В продакшене сюда нужно добавить оригинальную архитектуру из официального репо.

class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        # dummy for structure - replace with full implementation
        self.dummy = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.dummy(x), None, None, None, None, None, None
