import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# TokenEmbedding 将输入的每个单词（或称为token）转换成固定维度的向量 比如bert就是全变成768维的向量
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2  # for backward compatibility
        # tokenConv是一个卷积层，用于将输入的每个单词（或称为token）转换成固定维度的向量
        # in_channels是输入的通道数，out_channels是输出的通道数，kernel_size是卷积核的大小 padding是填充的大小 padding_mode是填充的模式 bias是是否使用偏置
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d): # 如果是卷积层
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu') # 使用kaiming正态分布初始化权重

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # [Batch, Time, Channel]
        return x

# 固定编码
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()  # 创建一个c_in * d_model的全零张量w
        w.require_grad = False  # 不需要梯度

        position = torch.arange(0, c_in).float().unsqueeze(1)   # 创建一个c_in * 1的张量position
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()     # 创建一个d_model/2 * 1的张量div_term 用于计算sin和cos

        w[:, 0::2] = torch.sin(position * div_term)     # 偶数列
        w[:, 1::2] = torch.cos(position * div_term)     # 奇数列

        self.emb = nn.Embedding(c_in, d_model)          # 创建一个c_in * d_model的embedding层
        self.emb.weight = nn.Parameter(w, requires_grad=False)      # 将w赋值给embedding层的权重

    def forward(self, x):
        return self.emb(x).detach() # 返回embedding层的输出 记得 detach

# 时序的编码
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # `day_size` 设置为 32，以表示一个月内所有可能的天数，包括天数少于 31 天的月份的占位符。
        # `month_size` 设置为 13，以包含所有 12 个月，外加一个额外的占位符，用于表示任何潜在的溢出或特殊情况。

        minute_size = 4 # 15min
        hour_size = 24  # 1h
        weekday_size = 7    # 一周
        day_size = 32   # 一个月
        month_size = 13 # 一年

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding # 固定编码或者embedding编码
        if freq == 't': # 15min
            self.minute_embed = Embed(minute_size, d_model) # 15min
        self.hour_embed = Embed(hour_size, d_model) # 1h
        self.weekday_embed = Embed(weekday_size, d_model)   # 一周
        self.day_embed = Embed(day_size, d_model)   # 一个月
        self.month_embed = Embed(month_size, d_model)   # 一年

    def forward(self, x):
        x = x.long()        # 转换为long类型
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.   # 如果有minute_embed的话就使用  没有的话就是0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x  # 返回所有的编码相加


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

# 数据embedding
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)   # tokenEmbedding
        self.position_embedding = PositionalEmbedding(d_model=d_model)  # 位置编码
        # 时序编码
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)    # dropout层

    def forward(self, x, x_mark):
        if x_mark is None:  # 如果没有时序编码 就只使用数值编码和位置编码
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

# Autoformer的数据embedding TimeMixer的数据embedding
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)      # 好像没有用到
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:  # 如果没有时序编码 就只使用数值编码
            x = self.value_embedding(x)
        else:   # 否则就使用数值编码和时序编码
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
