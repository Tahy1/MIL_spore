# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:15:35 2019
使用训练好的模型返回整张图片的概率矩阵
@author: SCSC
"""

import torch, time
from torch import nn
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference

def main():
    # 定义网络
    moment = time.time()
    model = resnet18.resnet18(False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('checkpoint_best.pth')['state_dict'])
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    # 定义数据集
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    test_dset = XDataset('test-100.lib',transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    # 定义log文件
    # 开始迭代
    test_dset.setmode(1)
    loss, probs = inference(0, test_loader, model, criterion)
    print(time.time() - moment)
    torch.save(probs, 'probs.pth')

if __name__ == '__main__':
    main()
