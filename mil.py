import torch, time, argparse, os, math
from random import shuffle as sf
from torch import nn
from torch import optim
import numpy as np
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err, tfpn, score, FocalLoss, BCE
'''

'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Validate')
    parser.add_argument('--norm', default='BN', type=str)
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--loss', default='CE', type=str)
    parser.add_argument('--pretrained', default='init', type=str)
    parser.add_argument('--size', type=str)
    parser.add_argument('--pk', type=int, default=1, help='number of positives')
    parser.add_argument('--nk', type=int, help='number of negatives')
    parser.add_argument('--gpu', type=int, help='index of GPU')
    parser.add_argument('--n_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--test_every', type=int, default=1, help='as you see')
    parser.add_argument('--ckpt', help='must enter your checkpoint path or none')
    args = parser.parse_args()
    return args

def main():
    fmoment = int(time.time())
    args = parse_args()
    norm = args.norm
    backbone = args.backbone
    pretrained = args.pretrained
    lossfunc = args.loss
    size = args.size
    pk = args.pk
    nk = args.nk
    n_epoch = args.n_epoch
    gpu = args.gpu
    test_every = args.test_every 
    ckpt = args.ckpt
    print('norm=%s backbone=%s pretrained=%s lossfunc=%s size=%s pk=%d nk=%d epoch=%d gpu=%d test_every=%d ckpt=%s'
          %(norm, backbone, pretrained, lossfunc, size, pk, nk, n_epoch, gpu, test_every, ckpt))
    if backbone =='resnet18':
        model = resnet18.resnet18(norm=norm).cuda(device=gpu)
    if pretrained == 'pretrained':
        ckpt_dict=torch.load('resnet18-pretrained.pth')
        model_dict = model.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict}
        model_dict.update(ckpt_dict)
        model.load_state_dict(model_dict)
    if lossfunc == 'CE':
        criterion = nn.CrossEntropyLoss().cuda(device=gpu)
    elif lossfunc == 'Focal':
        criterion = FocalLoss(class_num=2, gpu=gpu).cuda(device=gpu)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, -math.log(99))
    elif lossfunc == 'BCE':
        criterion = BCE(class_num=2, gpu=gpu).cuda(device=gpu)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cudnn.benchmark = True
    train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.2005,0.1490,0.1486],
                                                           std=[0.1445,0.1511,0.0967])])
    infer_trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.2005,0.1490,0.1486],
                                                           std=[0.1445,0.1511,0.0967])])
    train_dset = XDataset('train-%s.lib'%size, train_trans=train_trans, infer_trans=infer_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=64,shuffle=False,
        pin_memory=True)
    test_dset = XDataset('test-%s.lib'%size, train_trans=train_trans, infer_trans=infer_trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128,shuffle=False,
        pin_memory=True)
    
    if ckpt != 'none':
        checkpoint = torch.load(ckpt)
        start = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_f1 = checkpoint['best_f1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        if not os.path.exists('logs/Training_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment)):
            fconv = open('logs/Training_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment), 'w')
            fconv.write('time,epoch,loss,error\n')
            fconv.write('%d,0,0,0\n'%fmoment)
            fconv.close()
        if not os.path.exists('logs/Testing_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment)):
            fconv = open('logs/Testing_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment), 'w')
            fconv.write('time,epoch,loss,error,tp,tn,fp,fn,f1,S\n')
            fconv.write('%d,0,0,0\n'%fmoment)
            fconv.close()
    else:
        start = 0
        best_f1 = 0
        fconv = open('logs/Training_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment), 'w')
        fconv.write('time,epoch,loss,error\n')
        fconv.write('%d,0,0,0\n'%fmoment)
        fconv.close()
        
        fconv = open('logs/Testing_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment), 'w')
        fconv.write('time,epoch,loss,error,tp,tn,fp,fn,f1,S\n')
        fconv.write('%d,0,0,0\n'%fmoment)
        fconv.close()
    
    for epoch in range(start, n_epoch):
        train_dset.setmode(1)
        _, probs = inference(epoch, train_loader, model, criterion, gpu)
#        torch.save(probs,'probs/train-%d.pth'%(epoch+1))
        probs1 = probs[:train_dset.plen]
        probs0 = probs[train_dset.plen:]

        topk1 = np.array(group_argtopk(np.array(train_dset.slideIDX[:train_dset.plen]), probs1, pk))
        topk0 = np.array(group_argtopk(np.array(train_dset.slideIDX[train_dset.plen:]), probs0, nk))+train_dset.plen
        topk = np.append(topk1, topk0).tolist()
#        torch.save(topk,'topk/train-%d.pth'%(epoch+1))
#        maxs = group_max(np.array(train_dset.slideIDX), probs, len(train_dset.targets))
#        torch.save(maxs, 'maxs/%d.pth'%(epoch+1))
        sf(topk)
        train_dset.maketraindata(topk)
        train_dset.setmode(2)
        loss, err = train(train_loader, model, criterion, optimizer, gpu)
        moment = time.time()
        writecsv([moment, epoch+1, loss, err], 'logs/Training_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment))
        print('Training epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
        if (epoch+1) % test_every == 0:
            test_dset.setmode(1)
            loss, probs = inference(epoch, test_loader, model, criterion, gpu)
#            torch.save(probs,'probs/test-%d.pth'%(epoch+1))
#            topk = group_argtopk(np.array(test_dset.slideIDX), probs, pk)
#            torch.save(topk, 'topk/test-%d.pth'%(epoch+1))
            maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))  #è¿åæ¯ä¸ªåççæå¤§æ?ç
#            torch.save(maxs, 'maxs/test-%d.pth'%(epoch+1))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            tp, tn, fp, fn = tfpn(pred, test_dset.targets)
            err = calc_err(pred, test_dset.targets)
            S, f1 = score(tp, tn, fp, fn)
            moment = time.time()
            writecsv([moment, epoch+1, loss, err, tp, tn, fp, fn, f1, S], 'logs/Testing_%s_%s_%s_%s_%s_%d_%d_%d.csv'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment))
            print('Testing epoch=%d, loss=%.5f, error=%.5f'%(epoch+1, loss, err))
            #Save best model
            if f1 >= best_f1:
                best_f1 = f1
                obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_f1': best_f1,
                        'optimizer' : optimizer.state_dict()
                        }
                torch.save(obj, 'ckpt_%s_%s_%s_%s_%s_%d_%d_%d.pth'%(norm,backbone,pretrained,lossfunc,size,pk,nk,fmoment))
            
if __name__ == '__main__':
    main()
