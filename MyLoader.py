# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:59:53 2019

@author: SCSC
"""

import torch
import sys
from PIL import Image
import torch.utils.data as data

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        size = lib['size'] // 2
        tiles = []
        slideIDX = []
        plen = 0
        self.targets = lib['targets']
        for i,name in enumerate(lib['slides']):
            slide = Image.open('RGB/%s.jpg'%name)
            sys.stdout.write('Cutting JPGs: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            grids = lib['grid'][i]
            for grid in grids:
                tiles.append(transform(slide.crop((grid[1]-size,grid[0]-size,grid[1]+size,grid[0]+size))))
            slideIDX.extend([i]*len(grids))
            if self.targets[i] == 1:
                plen += len(grids)
        print('')
        self.tiles = tiles
        self.slideIDX = slideIDX
        self.plen = plen
        print('Number of tiles:%d'%len(slideIDX))
        self.mode = 1
        
    def setmode(self, m):
        self.mode = m
    def maketraindata(self, k):
        self.t_data = [(self.slideIDX[x], self.tiles[x], self.targets[self.slideIDX[x]]) for x in k]
    def __getitem__(self,index):
        if self.mode == 1:
            return self.tiles[index], self.targets[self.slideIDX[index]]
        elif self.mode == 2:
            slideIDX, tiles, targets = self.t_data[index]
            return tiles, targets
    def __len__(self):
        if self.mode == 1:
            return len(self.slideIDX)
        elif self.mode == 2:
            return len(self.t_data)