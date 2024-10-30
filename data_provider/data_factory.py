# data_factory.py 数据的工厂模式

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}

# data_provider 数据提供 所以是根据任务类型返回不同的数据集和dataloader
def data_provider(args, flag):
    Data = data_dict[args.data]     # 数据集类型 从data_dict中获取
    timeenc = 0 if args.embed != 'timeF' else 1     # 时间编码 0: 不使用时间编码 1: 使用时间编码

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True    # 是否打乱数据集 测试集不打乱
    drop_last = False   # 是否丢弃最后一个batch
    batch_size = args.batch_size    # batch_size
    freq = args.freq    # 数据的采集频率 15min 30min 1h 1d

    if args.task_name == 'anomaly_detection':   # 异常检测
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,  # 窗口大小 等于seq_len
            flag=flag,  # 数据集标志 train test
        )
        print(flag, len(data_set))
        data_loader = DataLoader(   # 创建dataloader
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification': # 分类任务
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:   # 其他任务 包括长期预测 短期预测 插值
        if args.data == 'm4':   # m4数据集（用于短期预测
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
