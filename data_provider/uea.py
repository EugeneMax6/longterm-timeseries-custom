import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    # 这个函数的作用是将data中的数据转换为模型需要的数据格式
    # data是一个list，其中每个元素是一个tuple，tuple中包含两个元素，第一个元素是X，第二个元素是y
    # X是一个torch tensor，shape是(seq_length, feat_dim)，seq_length是变长的
    # y是一个torch tensor，shape是(num_labels,)，num_labels > 1，用于多任务模型
    # max_len是全局固定的序列长度，用于需要固定长度输入的架构，其中batch长度不能动态变化
    # 较长的序列被剪切，较短的序列被填充为0
    # 返回X，targets，target_masks，padding_masks
    # X的shape是(batch_size, padded_length, feat_dim)，填充后的特征
    # targets的shape是(batch_size, padded_length, feat_dim)，未填充的特征
    # target_masks的shape是(batch_size, padded_length, feat_dim)，布尔值，0表示掩码值，1表示未受影响的特征值
    # padding_masks的shape是(batch_size, padded_length)，布尔值，1表示在这个位置保留向量，0表示填充

    batch_size = len(data)  # batch_size是data的长度
    features, labels = zip(*data)   # 将data中的数据分别存储到features和labels中 zip(*)是解压缩

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series   # 每个时间序列的原始序列长度
    if max_len is None:
        max_len = max(lengths)  # 未指定的话 max_len是lengths中的最大值

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)  # clip to max_len if sequence is too long
        X[i, :end, :] = features[i][:end, :]    # 将features中的数据填充到X中

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels) stack将多个tensor堆叠在一起
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    # 用于掩盖填充位置：从序列长度的张量中创建一个(batch_size, max_len)布尔掩码，其中1表示保留该位置的元素（时间步）
    # lengths是一个张量，表示每个时间序列的长度
    # max_len是最大长度，未指定的话是lengths的最大值

    batch_size = lengths.numel()    # numel()返回tensor中元素的个数
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    # 下面返回一个(batch_size, max_len)的张量，其中1表示保留该位置的元素（时间步）
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
