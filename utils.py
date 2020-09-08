import csv, torch, sys
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

def writecsv(wlist, dst):
    wlist = list(map(str,wlist))
    with open(dst,'a', newline='') as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(wlist)

def inference(run, loader, model, criterion, gpu):
    model.eval()
    running_loss = 0.
    probs = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            sys.stdout.write('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]\r'.format(run+1, 300, i+1, len(loader)))
            sys.stdout.flush()
            target = target.cuda(device=gpu, non_blocking=True)
            input = input.cuda(device=gpu, non_blocking=True)
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item()*input.size(0)
            output = F.softmax(output, dim=1)
            probs.extend(output.detach()[:,1].cpu().numpy().tolist()) #输出的第一列（预测值为正例则输出1，反例输出0）放到probs里
    print('')
    probs = np.array(probs)
    loss = running_loss/len(loader.dataset)
    return loss, probs

def train(loader, model, criterion, optimizer, gpu):
    model.train()
    running_loss = 0.
    probs = []
    real = []
    for i, (input, target) in enumerate(loader):
        real.extend(target.numpy().tolist())
        input = input.cuda(device=gpu, non_blocking=True)
        target = target.cuda(device=gpu, non_blocking=True)
        output = model(input)
        pred = F.softmax(output, dim=1)
        probs.extend(pred.detach()[:,1].cpu().numpy().tolist())
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    probs = [1 if x >= 0.5 else 0 for x in probs]
    err = calc_err(probs, real)
    loss = running_loss/len(loader.dataset)
    return loss, err

def calc_err(probs,real):
    probs = np.array(probs)
    real = np.array(real)
    assert len(probs) == len(real)
    neq = np.not_equal(probs, real)
    err = float(neq.sum())/probs.shape[0]
#    fpr = float(np.logical_and(probs==1,neq).sum())/(real==0).sum()  #FP占所有负例的比例
#    fnr = float(np.logical_and(probs==0,neq).sum())/(real==1).sum()  #FN占所有正例的比例
    return err
#    return err, fpr, fnr

def tfpn(probs, real):
    probs = np.array(probs)
    real = np.array(real)
    assert len(probs) == len(real)
    [[tn, fp], [fn, tp]] = confusion_matrix(real, probs)
    return tp, tn, fp, fn

def score(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    S = (tp + f1) / (tp + fn + 1)
    return S, f1
    

def group_argtopk(groups, data,k=1):  #groups为所有瓦片对应的切片序号组成的array，data为这些瓦片的预测值
    order = np.lexsort((data, groups))  #首先按照groups的元素排序，如果出现相同大小的情况，则再按照data排序，由小到大。
    groups = groups[order]  #把order对应的groups元素取出来
    data = data[order]  #把排好序的data取出来
    index = np.empty(len(groups), 'bool')
    index[-k:] = True  #最后K个设为True
    index[:-k] = groups[k:] != groups[:-k]  #将groups错开k位，这样就能保证每个slides返回概率最大的k个tiles
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]  #由Numpy的填充性质可得，可以保证被抽到的groups都有该groups最大的data(概率)，同时最大的概率对应的group一定会存在于out中。
    return out

class FocalLoss(nn.Module):
    def __init__(self, class_num, gpu=None, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.gpu = gpu

    def forward(self, inputs, targets):
        targets = targets.reshape(targets.shape[0],1)
        onehots = torch.zeros(targets.shape[0], self.class_num)
        m = nn.Sigmoid()
        if self.gpu != None:
            self.alpha = self.alpha.cuda(device=self.gpu)
            onehots = onehots.cuda(device=self.gpu)
        onehots = torch.scatter(onehots, 1, targets, 1)
        P = m(inputs)
        loss = - (1 - P) ** self.gamma * onehots * torch.log(P) - \
            P ** self.gamma * (1 - onehots) * torch.log(1 - P)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class BCE(nn.Module):
    def __init__(self, class_num, gpu=None):
        super(BCE, self).__init__()
        self.class_num = class_num
        self.gpu = gpu
    def forward(self, inputs, targets):
        loss = nn.BCELoss()
        m = nn.Sigmoid()
        targets = targets.reshape(targets.shape[0],1)
        onehots = torch.zeros(targets.shape[0], self.class_num)
        if self.gpu != None:
            onehots = onehots.cuda(device=self.gpu)
        onehots = torch.scatter(onehots, 1, targets, 1)
        output = loss(m(inputs), onehots)
        return output
