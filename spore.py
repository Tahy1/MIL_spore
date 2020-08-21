import torch, time, argparse, os
from random import shuffle as sf
from torch import nn
from torch import optim
import numpy as np
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err
'''
经测试，batchsize的大小对inference时间不会造成影响，所以选用128
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Validate')
    parser.add_argument('--pk', type=int, default=1, help='number of positives')
    parser.add_argument('--nk', type=int, help='number of negatives')
    parser.add_argument('--gpu', type=int, help='index of GPU')
    parser.add_argument('--n_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--test_every', type=int, default=1, help='as you see')
    parser.add_argument('--ckpt', help='must enter your checkpoint path or none')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pk = args.pk #选取的正例数目
    nk = args.nk #选取的反例数目
    n_epoch = args.n_epoch #迭代次数
    gpu = args.gpu
    test_every = args.test_every #训练n次测试一次
    ckpt = args.ckpt
    print('pk=%d nk=%d epoch=%d gpu=%d test_every=%d ckpt=%s'%(pk, nk, n_epoch, gpu, test_every, ckpt))
    # 定义网络
    model = resnet18.resnet18(False).cuda(device=gpu)
    #model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
    criterion = nn.CrossEntropyLoss().cuda(device=gpu)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cudnn.benchmark = True
    # 定义数据集
    train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ColorJitter(brightness=[0.4,1.3],
                                                             contrast=[0.7,1.8],
                                                             saturation=[0.6,1.7],
                                                             hue=[-0.1,0.03]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5,0.5,0.5],
                                                           std=[0.1,0.1,0.1])])
    infer_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5,0.5,0.5],
                                                           std=[0.1,0.1,0.1])])
    train_dset = XDataset('train-100.lib', train_trans=train_trans, infer_trans=infer_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=256,shuffle=False,
        pin_memory=True)
    test_dset = XDataset('test-100.lib', train_trans=train_trans, infer_trans=infer_trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=256,shuffle=False,
        pin_memory=True)
    
    if ckpt != 'none':
        #model_dict = model.state_dict()
        #ckpt = torch.load(ckpt)
        #ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
        #model_dict.update(ckpt)
        #model.load_state_dict(model_dict)
        checkpoint = torch.load(ckpt)
        start = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start = 0
        best_acc = 0

    if os.path.exists('Training_%d_%d.csv'%(pk,nk)):
        s = 'Warning!\nThere are several related logs '+\
              'or checkpoints exist in this path, make sure'+\
                  'you wanna overwrite them, or you will lost them.\n'+\
                      'enter YES to continue, others will exit.\n'
        choise = input(s)
        while choise == '':
            choise = input()
        if choise != 'YES':
            raise Exception('You enter others.')
    # 定义log文件
    fconv = open('Training_%d_%d.csv'%(pk,nk), 'w')
    fconv.write('time,epoch,loss,error\n')
    moment = time.time()
    fconv.write('%d,0,0,0\n'%moment)
    fconv.close()
    
    fconv = open('Testing_%d_%d.csv'%(pk,nk), 'w')
    fconv.write('time,epoch,loss,error\n')
    moment = time.time()
    fconv.write('%d,0,0,0\n'%moment)
    fconv.close()
    
    # 开始迭代
    for epoch in range(start, n_epoch):
        ## ①全部检测
        train_dset.setmode(1)
        _, probs = inference(epoch, train_loader, model, criterion, gpu)
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
        loss, err = train(train_loader, model, criterion, optimizer, gpu)
        moment = time.time()
        writecsv([moment, epoch+1, loss, err], 'Training_%d_%d.csv'%(pk,nk))
        print('Training epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
        ## ⑤验证
        if (epoch+1) % test_every == 0:
            test_dset.setmode(1)
            loss, probs = inference(epoch, test_loader, model, criterion, gpu)
#            torch.save(probs,'probs/test-%d.pth'%(epoch+1))
#            topk = group_argtopk(np.array(test_dset.slideIDX), probs, pk)
#            torch.save(topk, 'topk/test-%d.pth'%(epoch+1))
            maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))  #返回每个切片的最大概率
#            torch.save(maxs, 'maxs/test-%d.pth'%(epoch+1))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err = calc_err(pred, test_dset.targets)
            moment = time.time()
            writecsv([moment, epoch+1, loss, err], 'Testing_%d_%d.csv'%(pk,nk))
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
                torch.save(obj, 'checkpoint_best_%d_%d.pth'%(pk,nk))
            
if __name__ == '__main__':
    main()
