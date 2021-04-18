from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import Softmax
import torch
import torch.nn as nn


__call__ = ['CrossEntropy','BCEWithLogLoss','CrossEntropyWithLogLoss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss

class CrossEntropyWithLogLoss(object):
    def __init__(self, num_classes):
        self.loss_fn = CrossEntropyLoss()
        self.softmax = Softmax(dim=1)
        self.num_classes = num_classes

    def __call__(self,output,target):
        # output -> batch_size, num_classes*num_attributes
        output = output.float()

        # output -> batch_size, num_classes, num_attributes
        output = torch.reshape(output, (output.size()[0], output.size()[1] // self.num_classes, self.num_classes))
        output = torch.transpose(output, 1, 2)

        loss = self.loss_fn(input = output,target = target)
        return loss

class MSELoss(object):
    def __init__(self):
        self.loss_fn = nn.MSELoss()
    
    def __call__(self,output,target):
        loss = self.loss_fn(input=output.float(), target=target.float())
        return loss



