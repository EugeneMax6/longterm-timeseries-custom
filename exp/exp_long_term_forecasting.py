# exp/exp_long_term_forecasting.py 长期预测的任务代码

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

# Long Term Forecasting
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self): # build model
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:   # multi-gpu
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):  # get data
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):    # 优化器 Adam
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)       # Adam优化器
        return model_optim

    def _select_criterion(self):    # 损失函数 MSE
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):  # 用于验证 评估模型
        total_loss = []
        self.model.eval()   # 模型评估模式
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):    # 取batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)     # x的时间编码
                batch_y_mark = batch_y_mark.float().to(self.device)     # y的时间编码

                # decoder input 解码器输入
                # 在Informer中，为了避免step-by-step的解码结构，作者直接将x_enc中后label_len个时刻的数据和要预测时刻的数据进行拼接得到解码器输入。
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()   # 预测长度的0
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # 拼接 label_len长度 和 pred_len 的 0 序列
                # encoder - decoder
                if self.args.use_amp:   # 混合精度训练
                    with torch.cuda.amp.autocast(): # 自动混合精度
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 进入模型的输入是 batch_x, batch_x_mark, dec_inp, batch_y_mark 分别是 x, x的时间编码, 解码器输入, y的时间编码
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0  # 特征维度 -1: 多维特征 0: 单维特征
                outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 取预测长度的输出
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 取预测长度的真实值

                pred = outputs.detach().cpu()   # 预测值
                true = batch_y.detach().cpu()   # 真实值

                loss = criterion(pred, true)

                total_loss.append(loss) # 损失 append 进去
        total_loss = np.average(total_loss) # 平均损失
        self.model.train()
        return total_loss

    def train(self, setting):   # 训练
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):    # 创建文件夹
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader) # 训练步数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)   # 早停 patience 连续多少次没有提升就停止

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:   # 混合精度训练
            scaler = torch.cuda.amp.GradScaler()    # scaler 是一个梯度缩放器 用于混合精度训练

        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 迭代次数
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1 # 迭代次数 + 1
                model_optim.zero_grad()     # 梯度清零
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input 同上面的valid
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder 同上面valid一样的 过模型一遍
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0     # 特征维度 -1: 多维特征 0: 单维特征
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]      # 取预测长度的输出
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 取预测长度的真实值
                        loss = criterion(outputs, batch_y)  # 计算损失
                        train_loss.append(loss.item())  # 训练损失 append 进去
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # 每迭代 100 次输出一次信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                # 梯度下降
                if self.args.use_amp: # 混合精度训练
                    scaler.scale(loss).backward()   # 反向传播
                    scaler.step(model_optim)        # 更新参数
                    scaler.update()            # 更新scaler
                else:   # 普通训练
                    loss.backward()    # 反向传播
                    model_optim.step()      # 更新参数

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)    # 验证集损失
            test_loss = self.vali(test_data, test_loader, criterion)    # 测试集损失

            # 输出 训练集损失 验证集损失 测试集损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)   # 早停 保存模型的代码在早停里面
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)     # 调整学习率

        best_model_path = path + '/' + 'checkpoint.pth'   # 最好的模型路径
        self.model.load_state_dict(torch.load(best_model_path)) # 加载最好的模型

        return self.model

    # test 也是 类似
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))  # 加载模型

        # preds 是 预测值 trues 是 真实值
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' # test结果保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0 # 特征维度  -1: 多维特征 0: 单维特征
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                # 用detach()将outputs从计算图中分离出来，不再计算outputs的梯度
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:   # 反标准化
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                # 取 特征维度
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs  # 预测值
                true = batch_y  # 真实值

                preds.append(pred)
                trues.append(true)
                # 每 20 次保存一次结果并可视化
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)  # 反标准化 + reshape
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)  # 真实值 ground truth
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)  # 预测值 prediction
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))  # 可视化

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # 两次reshape 为了计算dtw
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation dtw是动态时间规整 用于时间序列的相似度计算
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)     # 曼哈顿距离 差值的绝对值
            for i in range(preds.shape[0]):
                # 两次reshape 为了计算dtw  (batch_size, seq_len, features) -> (seq_len, features)
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean() # 计算dtw的均值
        else:
            dtw = 'not calculated'
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        # 输出写文件里面
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
