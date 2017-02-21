###############################
## This document created by Alexandre Boulch, ONERA, France is
## distributed under GPL license
###############################

import numpy as np
import argparse
import os
from random import  shuffle
from tqdm import *

##########
# TORCH
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#########

import segnet

input_nbr = 3
imsize = 224

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SegNet example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

# cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
USE_CUDA = args.cuda

# set the seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Create SegNet model
label_nbr = 45
model = SegNet(label_nbr)
model.load_weights("vgg16-00b39a1b.pth") # load segnet weights
if USE_CUDA:# convert to cuda if needed
    model.cuda()
else:
    model.float()
model.eval()
print(model)


# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()

    # update learning rate
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # define a weighted loss (0 weight for 0 label)
    weights_list = [0]+[1 for i in range(17)]
    weights = np.asarray(weights_list)
    weigthtorch = torch.Tensor(weights_list)
    if(USE_CUDA):
        loss = nn.CrossEntropyLoss(weight=weigthtorch).cuda()
    else:
        loss = nn.CrossEntropyLoss(weight=weigthtorch)


    total_loss = 0

    # iteration over the batches
    batches = []
    for batch_idx,batch_files in enumerate(tqdm(batches)):

        # containers
        batch = np.zeros((args.batch_size,input_nbr, imsize, imsize), dtype=float)
        batch_labels = np.zeros((args.batch_size,imsize, imsize), dtype=int)

        # fill the batch
        # ...

        batch_th = Variable(torch.Tensor(batch))
        target_th = Variable(torch.LongTensor(batch_labels))

        if USE_CUDA:
            batch_th =batch_th.cuda()
            target_th = target_th.cuda()

        # initilize gradients
        optimizer.zero_grad()

        # predictions
        output = model(batch_th)

        # Loss
        output = output.view(output.size(0),output.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)

        l_ = loss(output.cuda(), target)
        total_loss += l_.cpu().data.numpy()
        l_.cuda()
        l_.backward()
        optimizer.step()

    return total_loss/len(files)

def test(epoch):
    model.eval()

    # iteration over the batches
    batches = []
    for batch_idx,batch_files in enumerate(tqdm(batches)):

        # containers
        bs = len(batch_files)
        batch = np.zeros((bs,input_nbr, imsize, imsize), dtype=float)
        batch_labels = np.zeros((bs,imsize, imsize), dtype=int)

        # fill batches
        # ...

        data_s2 = Variable(torch.Tensor(batch))
        target = Variable(torch.LongTensor(batch_labels))
        if USE_CUDA:
            data_s2, target = data_s2.cuda(), target.cuda()

        batch_th = Variable(torch.Tensor(batch))
        target_th = Variable(torch.LongTensor(batch_labels))

        if USE_CUDA:
            batch_th =batch_th.cuda()
            target_th = target_th.cuda()

        # predictions
        output = model(batch_th)

        # ...

for epoch in range(1, args.epochs + 1):
    print(epoch)

    # training
    train_loss = train(epoch)
    print("train_loss "+str(train_loss))

    # validation / test
    test(epoch)
