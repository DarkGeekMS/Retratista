from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import Softmax
import torch
import torch.nn as nn

class MSELoss(object):
    def __init__(self):
        self.loss_fn = nn.MSELoss()
    
    def __call__(self,output,target):
        loss = self.loss_fn(input=output.float(), target=target.float())
        return loss