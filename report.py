import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

probs = torch.load('probs.pth')
maxs = []
for i in range(574):
    maxs.append(max(probs[i*713:(i+1)*713]))
targets = torch.load('test-100.lib')['targets']
preds = [1 if i>0.5 else 0 for i in maxs]
print(confusion_matrix(targets, preds))
print(classification_report(targets, preds, digits=4))
print('AUC:',roc_auc_score(targets,maxs))
