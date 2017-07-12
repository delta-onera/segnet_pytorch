from __future__ import print_function
import os
import os.path
import sys
import random
import time

import numpy as np
import PIL
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch.autograd
import torch.autograd.variable
import torchvision
import torchvision.transforms

import csv

print("init")
pathToModel = "PUT_HERE_THE_PATH_TO_VGG_MODEL/vgg16-00b39a1b.pth"
if not torch.cuda.is_available():
    print('WTF no cuda')
    quit()
torch.cuda.manual_seed(1)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 =    nn.Conv2d(3, 64, kernel_size=3,padding=1, bias=True)
        self.conv1_2 =   nn.Conv2d(64, 64, kernel_size=3,padding=1, bias=True)
        self.conv2_1 =  nn.Conv2d(64, 128, kernel_size=3,padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3,padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3,padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3,padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.prob = nn.Conv2d(512, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        
        x = self.prob(x)
        return x

    def load_weights(self, model_path):
        correspondance=[]
        correspondance.append(("features.0","conv1_1"))
        correspondance.append(("features.2","conv1_2"))
        correspondance.append(("features.5","conv2_1"))
        correspondance.append(("features.7","conv2_2"))
        correspondance.append(("features.10","conv3_1"))
        correspondance.append(("features.12","conv3_2"))
        correspondance.append(("features.14","conv3_3"))
        correspondance.append(("features.17","conv4_1"))
        correspondance.append(("features.19","conv4_2"))
        correspondance.append(("features.21","conv4_3"))
        correspondance.append(("features.24","conv5_1"))
        correspondance.append(("features.26","conv5_2"))
        correspondance.append(("features.28","conv5_3"))
        
        model_dict = self.state_dict()
        pretrained_dict = torch.load(model_path)    
        
        for name1,name2 in correspondance:
            fw = False
            fb = False
            for name, param in pretrained_dict.items():
                if name==name1+".weight" :
                    model_dict[name2+".weight"].copy_(param)
                    fw=True
                if name==name1+".bias" :
                    model_dict[name2+".bias"].copy_(param)
                    fb=True
            if not fw:
                print(name2+".weight not found")
            if not fb:
                print(name2+".bias not found")
        self.load_state_dict(model_dict)

vggpooling = 16
imagemax = 256
        
vgg = VGG()
vgg.load_weights(pathToModel)
vgg.cuda()
vgg.train()

print("load data")
pathToDataset = "PUT_HERE_THE_PATH_TO_DATA"
imagename = os.listdir(pathToDataset+"/images")

vts = []
for name in imagename:
    vtfile = open(pathToDataset+"/detections/"+name+".txt", "rt")
    datalistcsv = csv.reader(vtfile,delimiter=" ")
    vtl = list(datalistcsv)
    vt = []
    for i in range(len(vtl)):
        vt.append((int(vtl[i][0]),int(vtl[i][1])))
    vts.append(vt)

print("training")
lr = 0.001
momentum = 0.5
optimizer = optim.SGD(vgg.parameters(), lr=lr, momentum=momentum)
losslayer = nn.CrossEntropyLoss()
lossduringtraining = open("build/lossduringtraining.txt","w")

nbepoch = 30

for epoch in range(nbepoch):
    for frame in range(len(imagename)):
        imagepil = PIL.Image.open(pathToDataset+"/images/"+imagename[frame]).copy()
        imagenp3 = np.asarray(imagepil)
        (w,h,_) = imagenp3.shape
        if w<imagemax or h<imagemax:
            print("w<imagemax or h<imagemax is not handled "+str(frame))
            continue

        offxy=[]
        for offy in range(0,h-imagemax+1,imagemax):
            for offx in range(0,w-imagemax+1,imagemax):
                offxy.append((offy,offx))
        random.shuffle(offxy)
        for offy,offx in offxy:
            crop = np.zeros((1,3,imagemax,imagemax), dtype=float)
            cropvt=[]
            for r in range(imagemax):
                for c in range(imagemax):
                    for ch in range(3):
                        crop[0][2-ch][r][c] = imagenp3[r+offy][c+offx][ch]/16
            for (x,y) in vts[frame]:
                if 0<=x-offx and x-offx<imagemax and 0<=y-offy and y-offy<imagemax:
                    cropvt.append((x-offx,y-offy))
            
            if len(cropvt)==0 and epoch==0:
                continue
            
            inputtensor = torch.autograd.Variable(torch.Tensor(crop))
            inputtensor = inputtensor.cuda()
            optimizer.zero_grad()
            
            outputtensor = vgg(inputtensor)
            
            desiredmask = np.zeros((1,1,imagemax//vggpooling,imagemax//vggpooling), dtype=float)
            for (x,y) in cropvt:
                desiredmask[0][0][y//vggpooling][x//vggpooling]=1
                if vggpooling/2<=x and x+vggpooling/2<imagemax and vggpooling/2<=y and y+vggpooling/2<imagemax :
                    desiredmask[0][0][(y+vggpooling//2)//vggpooling][x//vggpooling]=1
                    desiredmask[0][0][(y-vggpooling//2)//vggpooling][x//vggpooling]=1
                    desiredmask[0][0][y//vggpooling][(x+vggpooling//2)//vggpooling]=1
                    desiredmask[0][0][y//vggpooling][(x-vggpooling//2)//vggpooling]=1
            targettensor = torch.from_numpy(desiredmask).long()        
            targettensor = torch.autograd.Variable(targettensor)
            targettensor = targettensor.cuda()
            
            outputtensor = outputtensor.view(outputtensor.size(0),outputtensor.size(1), -1)
            outputtensor = torch.transpose(outputtensor,1,2).contiguous()
            outputtensor = outputtensor.view(-1,outputtensor.size(2))
            targettensor = targettensor.view(-1)
            loss = losslayer(outputtensor, targettensor)
            loss.backward()
            optimizer.step()
            
            if random.randint(0,10)==0:
                lossduringtraining.write(str(loss.cpu().data.numpy()[0]*16*16)+"\n")
                lossduringtraining.flush()

print("testing")
for name in imagename:
    pred = open("build/pred/"+name+".txt","w")
    imagepil = PIL.Image.open(pathToDataset+"/images/"+name).copy()
    imagenp3 = np.asarray(imagepil)
    (w,h,_) = imagenp3.shape
    if w<imagemax or h<imagemax:
        print("w<imagemax or h<imagemax is is not handled "+str(frame))
        continue

    offxy=[]
    for offy in range(0,h-imagemax+1,imagemax):
        for offx in range(0,w-imagemax+1,imagemax):
            offxy.append((offy,offx))
    
    for offy,offx in offxy:
        crop = np.zeros((1,3,imagemax,imagemax), dtype=float)
        cropvt=[]
        for r in range(imagemax):
            for c in range(imagemax):
                for ch in range(3):
                    crop[0][2-ch][r][c] = imagenp3[r+offy][c+offx][ch]/16
            
        inputtensor = torch.autograd.Variable(torch.Tensor(crop))
        inputtensor = inputtensor.cuda()
        
        outputtensor = vgg(inputtensor)
        
        proba = outputtensor.cpu().data.numpy()
        for r in range(imagemax//vggpooling):
            for c in range(imagemax//vggpooling):
                pred.write(str(vggpooling*c+offx)+" "+str(vggpooling*r+offy)+" "+str(proba[0][1][r][c]-proba[0][0][r][c])+"\n")
                pred.flush()
    
