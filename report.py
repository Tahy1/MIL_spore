# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:13:48 2019

@author: SCSC
"""

import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

probs = torch.load('maxs-1.pth')
targets = torch.load('test.lib')['targets']
preds = [1 if i>0.5 else 0 for i in probs]
print(confusion_matrix(targets, preds))
print(classification_report(targets, preds, digits=4))
print('AUC:',roc_auc_score(targets,probs))