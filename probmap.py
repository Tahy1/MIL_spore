# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:15:35 2019
使用训练好的模型返回整张图片的概率矩阵
@author: SCSC
"""

import torch
from random import shuffle as sf
from torch import nn
from torch import optim
import numpy as np
from MyLoader import MILdataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err
'''
经测试，batchsize的大小对inference时间不会造成影响，所以选用128
但是得益于大内存的优点，我们将所有数据提前切好，并且转换成Tensor存在内存中，随用随取，避免了重复切割
'''

def main():
    # 定义网络
    model = resnet18.resnet18(False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('checkpoint_best.pth')['state_dict'])
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    # 定义数据集
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    test_dset = MILdataset('test.lib',transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    # 定义log文件
    # 开始迭代
    test_dset.setmode(1)
    loss, probs = inference(0, test_loader, model, criterion)
    torch.save(probs, 'probs.pth')

            
main()