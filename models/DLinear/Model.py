# DLinear model adpated from the original implementation in the paper "Are Transformers Effective for Time Series Forecasting?" by Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. https://github.com/vivva/DLinear/blob/main/models/DLinear.py

from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.config import Config
from fastbound.bound_propagation import ibp_propagation

class Model(nn.Module):
    """
    DLinear - simplified
    """
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # Decompsition Kernel Size
        self.kernel_size = config.moving_avg
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
    
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)
        trend_init = self.avg(F.pad(x, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2), mode='replicate'))
        seasonal_init = x - trend_init

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1)
    
    def forward_bounds(self, x_lower, x_upper):
        x_lower, x_upper = x_lower.permute(0,2,1), x_upper.permute(0,2,1)
        trend_init_lower = self.avg(F.pad(x_lower, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2), mode='replicate'))
        trend_init_upper = self.avg(F.pad(x_upper, ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2), mode='replicate'))
        seasonal_init_lower = x_lower - trend_init_upper
        seasonal_init_upper = x_upper - trend_init_lower

        seasonal_output_lower, seasonal_output_upper = ibp_propagation(seasonal_init_lower, seasonal_init_upper, self.Linear_Seasonal)
        trend_output_lower, trend_output_upper = ibp_propagation(trend_init_lower, trend_init_upper, self.Linear_Trend)
        x_lower = seasonal_output_lower + trend_output_lower
        x_upper = seasonal_output_upper + trend_output_upper
        return x_lower.permute(0,2,1), x_upper.permute(0,2,1)

    def __str__(self):
        return 'DLinear'
