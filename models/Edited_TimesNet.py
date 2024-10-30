# 尝试修改TimesNet模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

# 快速傅里叶的部分
def FFT_for_Period(x, k=2):
    # k是超参数 取topK的频率
    # [B, T, C] means [Batch, Time, Channel]
    xf = torch.fft.rfft(x, dim=1)   # rfft是实数的傅里叶变换 dim=1表示对时间维度进行傅里叶变换
    # find period by amplitudes 通过振幅找周期
    frequency_list = abs(xf).mean(0).mean(-1)   # 取振幅的均值
    frequency_list[0] = 0   # 第一个频率为0
    _, top_list = torch.topk(frequency_list, k) # 取topK的频率 直接用torch.topk方法
    top_list = top_list.detach().cpu().numpy()  # 将top_list转换为numpy数组
    period = x.shape[1] // top_list  # 计算周期
    # 返回周期和权重 也就是振幅
    return period, abs(xf).mean(-1)[:, top_list]

# TimesBlock 模块 模型的关键部分
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  # 序列长度
        self.pred_len = configs.pred_len    # 预测长度
        self.k = configs.top_k  # topK
        # parameter-efficient design
        # TimesNet用到的Inception层 高效率的卷积设计
        self.conv = nn.Sequential(
            # Inception V1 Block
            # Inception_Block_V1(configs.d_model, configs.d_ff,
            #                    num_kernels=configs.num_kernels),

            # 用简单的2D卷积代替Inception_Block_V1
            nn.Conv2d(configs.d_model, configs.d_ff, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),

            nn.GELU(),  # gelu激活函数

            # Inception V1 Block
            # Inception_Block_V1(configs.d_ff, configs.d_model,
            #                    num_kernels=configs.num_kernels)

            # 2D卷积
            nn.Conv2d(configs.d_ff, configs.d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))


        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

# TimesNet模型
class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name  # 任务名称
        self.seq_len = configs.seq_len  # 序列长度
        self.label_len = configs.label_len  # 标签长度
        self.pred_len = configs.pred_len    # 预测长度
        # e_layers是encoder的层数 也就是TimesBlock的个数
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        # embedding层 enc_in是输入的特征维度 d_model是模型的维度 embed是嵌入维度 freq是频率 dropout是dropout
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers   # encoder的层数是模型层数  因为timesnet只用到了inception的卷积
        self.layer_norm = nn.LayerNorm(configs.d_model) # 归一化层 layernorm
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':   # 长/短期预测
            # 预测线性层 seq_len是序列长度 pred_len是预测长度
            # pred_len + seq_len是预测长度和序列长度的和 也就是输出的长度
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            # 最后的输出层
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection': # 插值和异常检测
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':  # 分类
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    # 预测函数 这里就用到了enc的部分 因为timesnet就只有卷积的部分 相当于没有decoder
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # 参考Non-stationary Transformers的归一化（？
        means = x_enc.mean(1, keepdim=True).detach()    # 计算均值
        x_enc = x_enc - means   # 减去均值
        # 标准差
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # 除以标准差得到encoder的输入

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] means [Batch, Time, Channel]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension 通过线形层对时间维度进行对齐
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))   # LayerNorm + TimesBlock
        # porject back  最后的输出层
        dec_out = self.projection(enc_out)

        # 反归一化
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':   # 长/短期预测
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)   # 预测函数
            return dec_out[:, -self.pred_len:, :]  # [B, L, D] means [Batch, Length, Dimension]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
