# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:32:56 2019

@author: SCSC
"""

import torch, sys
import numpy as np
probs = torch.load('probs.pth')
test = torch.load('test-100.lib')
grids = test['grid']
slides = test['slides']
targets = test['targets']

temp = []
for i in range(len(probs)//713):
    temp.append(probs[i*713:(i+1)*713])

heatmaps = []
for i in temp:
    t = []
    for j in range(23):
        t.append(i[31*j:31*(j+1)])
    heatmaps.append(t)
heatmaps = np.array(heatmaps)


import cv2
for i in range(len(slides)):
    sys.stdout.write('Processing: [{}/{}]\r'.format(i+1, len(slides)))
    sys.stdout.flush()
    img = cv2.imread('RGB/%s.jpg'%slides[i])
    heatmap = cv2.resize(heatmaps[i], (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap[heatmap!=255] = 0
    heatmap[:,:,2] = heatmap[:,:,0]
    heatmap[:,:,:2]=0
    superimposed_img = heatmap * 0.9 + img
    cv2.imwrite('results/%d_%s.jpg'%(targets[i], slides[i]), superimposed_img)


#import cv2
#temp = []
#for i in heatmaps:
#    t = []
#    for j in i:
#        t.extend(j)
#    temp.append(t)
#heatmaps = np.array(temp)
#for i in range(len(slides)):
#    img = cv2.imread('upload/%s.jpg'%slides[i])
#    for j in range(len(heatmaps[i])):
#        if heatmaps[i][j] >= 0.5:
#            cv2.rectangle(img,(grids[i][j][1]-50,grids[i][j][0]-50),(grids[i][j][1]+50,grids[i][j][0]+50),(0,0,255),2)
#    cv2.imwrite('rectangles/%d_%s.jpg'%(targets[i], slides[i]), img)
'''
import cv2
img = cv2.imread('upload/20180703182727167.jpg')
heatmap = cv2.resize(heatmaps[275], (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap[heatmap!=255] = 0
heatmap[:,:,2] = heatmap[:,:,0]
heatmap[:,:,:2]=0
superimposed_img = heatmap *0.8 + img
cv2.imwrite('temp.jpg', superimposed_img)
'''
