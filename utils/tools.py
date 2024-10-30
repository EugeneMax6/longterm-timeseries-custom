import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')

# 学习率策略
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':   # type1 是每隔1个epoch降低一次学习率
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8            # 降低的学习率是提前设定好的
        }
    elif args.lradj == "cosine":    # cosine 是cosine退火
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}    # 退火
    if epoch in lr_adjust.keys():   # 如果当前epoch在lr_adjust中
        lr = lr_adjust[epoch]   # 获取当前epoch对应的学习率
        for param_group in optimizer.param_groups:  # 更新所有参数组的学习率
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# early stop
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose      # 是否打印信息
        self.counter = 0
        self.best_score = None  # 记录最好的分数
        self.early_stop = False
        self.val_loss_min = np.Inf  # 记录最小的loss
        self.delta = delta  # delta是一个阈值，只有当loss减小大于这个阈值时才算作真正的减小

    # call方法是一个特殊方法，当一个类的实例被当做函数调用时会被调用
    def __call__(self, val_loss, model, path):
        score = -val_loss   # 评分标准
        if self.best_score is None: # 第一次调用
            self.best_score = score # 记录最好的分数
            self.save_checkpoint(val_loss, model, path) # 保存模型
        elif score < self.best_score + self.delta:  # 如果分数没有提升
            self.counter += 1   # 计数器加1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 输出信息 准备早停
            if self.counter >= self.patience:   # 超过 patience 就 early stop
                self.early_stop = True
        else:   # 分数提升 保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    # 保存checkpoints
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:    # 打印信息
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')   # 保存模型
        self.val_loss_min = val_loss    # 更新最小的loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""  # 通过.访问字典的属性
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# 数据标准化
class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# 结果可视化
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


# 异常检测的detection adjustment
def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

# 分类中用到的 计算准确率
def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)