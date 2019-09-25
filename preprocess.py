# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:56:20 2019
生成train.lib和test.lib
要求必须保证正例聚集在前，反例聚集在后
@author: SCSC
"""

import torch, os
import numpy as np
from PIL import Image
box = 100
positives = os.listdir('positives')
negatives = os.listdir('negatives')

trainlibrary = {}
testlibrary = {}

def makelist(typ):
    global box
    slides = []
    grid = []
    targets = []
    names = os.listdir(typ)
    for file in names:
        file = file.split('.')[0]
        data = np.array(Image.open('%s/%s.jpg'%(typ, file)))
        height, width= data.shape[:2]
        h_num = (height - box) / (box // 2) + 1
        hs = []
        for i in range(int(h_num)):
            hs.append((i+1)*(box // 2))
        if int(h_num) != h_num:
            hs.append(height-(box // 2))
        w_num = (width - box) / (box // 2) + 1
        ws = []
        for i in range(int(w_num)):
            ws.append((i+1)*(box // 2))
        if int(w_num) != w_num:
            ws.append(width-(box // 2))
        hws = []
        for h in hs:
            for w in ws:
                hws.append([h,w])
        grid.append(np.array(hws))
        slides.append(file)
        targets.append(1 if typ=='positives' else 0)
    return grid, slides, targets

pos_grid, pos_slides, pos_targets = makelist('positives')
neg_grid, neg_slides, neg_targets = makelist('negatives')

trainlibrary['targets'] = pos_targets[:int(len(positives)*0.7)] + neg_targets[:int(len(negatives)*0.7)]
trainlibrary['slides'] = pos_slides[:int(len(positives)*0.7)] + neg_slides[:int(len(negatives)*0.7)]
trainlibrary['grid'] = pos_grid[:int(len(positives)*0.7)] + neg_grid[:int(len(negatives)*0.7)]
trainlibrary['size'] = box
torch.save(trainlibrary, 'train-%d.lib'%box)

testlibrary['targets'] = pos_targets[int(len(positives)*0.7):] + neg_targets[int(len(negatives)*0.7):]
testlibrary['slides'] = pos_slides[int(len(positives)*0.7):] + neg_slides[int(len(negatives)*0.7):]
testlibrary['grid'] = pos_grid[int(len(positives)*0.7):] + neg_grid[int(len(negatives)*0.7):]
testlibrary['size'] = box
torch.save(testlibrary, 'test-%d.lib'%box)
