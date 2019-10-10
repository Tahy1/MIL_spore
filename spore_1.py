# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:39:17 2019

@author: SCSC
"""

import torch, time
from random import shuffle as sf
from torch import nn
from torch import optim
import numpy as np
from MyLoader import MyDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err
'''
经测试，batchsize的大小对inference时间不会造成影响，所以选用128
'''

def main():
    best_acc = 0
    pk = 1 #选取的正例数目
    nk = 5 #选取的反例数目
    n_epoch = 300 #迭代次数
    test_every = 1 #训练n次测试一次
    # 定义网络
    model = resnet18.resnet18(False)
    model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cudnn.benchmark = True
    # 定义数据集
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    train_dset = XDataset('train-100.lib',transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    test_dset = XDataset('test-100.lib',transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128,shuffle=False,
        pin_memory=False)
    # 定义log文件
    fconv = open('Training.csv', 'w')
    fconv.write('time,epoch,loss,error\n')
    moment = time.time()
    fconv.write('%d,0,0,0\n'%moment)
    fconv.close()
    
    fconv = open('Testing.csv', 'w')
    fconv.write('time,epoch,loss,error\n')
    moment = time.time()
    fconv.write('%d,0,0,0\n'%moment)
    fconv.close()
    # 开始迭代
    for epoch in range(n_epoch):
        ## ①全部检测
        train_dset.setmode(1)
        _, probs = inference(epoch, train_loader, model, criterion)
#        torch.save(probs,'probs/train-%d.pth'%(epoch+1))
        probs1 = probs[:train_dset.plen] #plen是指probs中来自正例的tiles(probs)数目
        probs0 = probs[train_dset.plen:]
        ## ②选出前pk=1个
        topk1 = np.array(group_argtopk(np.array(train_dset.slideIDX[:train_dset.plen]), probs1, pk))
        ## ②选出前nk=5个，并偏移plen个位置
        topk0 = np.array(group_argtopk(np.array(train_dset.slideIDX[train_dset.plen:]), probs0, nk))+train_dset.plen
        topk = np.append(topk1, topk0).tolist()
#        torch.save(topk,'topk/train-%d.pth'%(epoch+1))
#        maxs = group_max(np.array(train_dset.slideIDX), probs, len(train_dset.targets))
#        torch.save(maxs, 'maxs/%d.pth'%(epoch+1))
        sf(topk)
        ## ③准备训练集
        train_dset.maketraindata(topk)
        train_dset.setmode(2)
        ## ④训练并保存结果
        loss, err = train(train_loader, model, criterion, optimizer)
        moment = time.time()
        writecsv([moment, epoch+1, loss, err], 'Training.csv')
        print('Training epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
        ## ⑤验证
        if (epoch+1) % test_every == 0:
            test_dset.setmode(1)
            loss, probs = inference(epoch, test_loader, model, criterion)
#            torch.save(probs,'probs/test-%d.pth'%(epoch+1))
#            topk = group_argtopk(np.array(test_dset.slideIDX), probs, pk)
#            torch.save(topk, 'topk/test-%d.pth'%(epoch+1))
            maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))  #返回每个切片的最大概率
#            torch.save(maxs, 'maxs/test-%d.pth'%(epoch+1))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err = calc_err(pred, test_dset.targets)
            moment = time.time()
            writecsv([moment, epoch+1, loss, err], 'Testing.csv')
            print('Testing epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
            #Save best model
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict()
                        }
                torch.save(obj, 'checkpoint_best.pth')
            
if __name__ == '__main__':
    main()
