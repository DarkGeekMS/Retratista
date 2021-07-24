#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from utils.ddfa import DDFADataset, ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
import torchvision.models as models



snapshot="snapshot/model_pdc" 
param_fp_train='train.configs/param_all_norm.pkl' 
param_fp_val='train.configs/param_all_norm_val.pkl' 
warmup=5
batch_size=512
val_batch_size = 32
base_lr=0.01
epochs=50 
milestones=[30,40]
filelists_train="train.configs/train_aug_120x120.list.train"
filelists_val="train.configs/train_aug_120x120.list.val"
root="./train_aug_120x120"
num_classes = 62

lr = None


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= warmup:
            return 1
        elif warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    global lr
    lr = base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main():

    model = models.mobilenet_v2(pretrained=False)
    classifier_layers = list(model.classifier[:-1])
    output_layer = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    classifier_layers.append(output_layer)
    model.classifier = nn.Sequential(*classifier_layers)
    model.cuda()

    criterion = nn.MSELoss(size_average=True).cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=base_lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)


    normalize = NormalizeGjz(mean=127.5, std=128) 

    train_dataset = DDFADataset(root=root, filelists=filelists_train, param_fp=param_fp_train, transform=transforms.Compose([ToTensorGjz(), normalize]))
    val_dataset = DDFADataset(root=root,filelists=filelists_val,param_fp=param_fp_val,transform=transforms.Compose([ToTensorGjz(), normalize]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=8, shuffle=False)

    cudnn.benchmark = True

    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch, milestones)

        losses = AverageMeter()

        model.train()

        for i, (input, target) in enumerate(train_loader):

            input = input.cuda()

            target.requires_grad_(False)

            target = target.cuda()

            output = model(input)


            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if i % 50 == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t',f'LR: {lr:8f}\t', f'Loss {losses.val:.4f} ({losses.avg:.4f})')
                            
            filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
            state_dict =  {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                }
            torch.save(state_dict,filename)

        model.eval()

        with torch.no_grad():
            val_losses = []
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda()
                target.requires_grad_(False)
                target = target.cuda()
                output = model(input)

                val_loss = criterion(output, target)
                val_losses.append(val_loss.item())

            val_loss = np.mean(val_losses)
            print(f'Val: [{epoch}][{len(val_loader)}]\t', f'Loss {val_loss:.4f}\t')

if __name__ == '__main__':
    main()
