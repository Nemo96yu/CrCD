import torch
import torchvision.transforms as transforms
import ssl
import numpy as np
from scipy.io import loadmat
# from utils_HSI.utils import splitSampleByClass
# from torch.utils.data import DataLoader
# from utils_HSI.HSIDataset import HSIDatasetV1, DatasetInfo
import configs.configs as cfg
from utils_HSI.HSI_data import batch_collate as collate_fn
from utils_HSI.HSI_data import HSI_data as fun_data
from utils_HSI.get_train_test_set import get_train_test_set as fun_get_set
from torch.utils.data import Dataset
ssl._create_default_https_context = ssl._create_unverified_context


def load_Indian(partial_rate, batch_size):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # 基础参数配置
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']  # 导入cfg.train的'train_model'的一系列参数
    cfg_optim = cfg.train['optimizer']  # 导入cfg.train的'optimizer'的一系列参数
    cfg_test = cfg.test
    IS_train = True
    IS_test = False
    # 数据导入和数据集划分
    data_sets, partialY = fun_get_set(cfg_data)

    train_dataset = fun_data(data_sets, cfg_data['train_data'], IS_train, partial_rate=0.1)
    test_dataset = fun_data(data_sets, cfg_data['test_data'], IS_test, partial_rate=0.1)
    no_gt_data = fun_data(data_sets, cfg_data['no_gt_data'], IS_test, partial_rate=0.1)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    train_sampler = 0
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    return partial_matrix_train_loader, partialY, train_sampler, test_loader




