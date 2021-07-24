import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Deep speech 2 consists of 3 ResCNN followed by Linear Layer followed by 5 Bidirectional RNN followed by softmax before entering the CTC decoder
"""


class ResCNN(nn.Module):
    """
        Residual CNN 
    """
    def __init__(self, inChannels, outChannels, kernel, dropOut, features):
        super(ResCNN, self).__init__()

        self.CNN1 = nn.Conv2d(inChannels, outChannels, kernel, 1, padding=kernel//2)
        self.CNN2 = nn.Conv2d(outChannels, outChannels, kernel, 1, padding=kernel//2)
        self.dropOut = nn.Dropout(dropOut)
        self.layerNorm = nn.LayerNorm(features)
        self.layerNorm2 = nn.LayerNorm(features)

    def forward(self, x):
        residual = x  # keep the original input
        # transpose the msg to calculate the norm on the features

        x = x.transpose(2,3).contiguous() # result = (batch, channel, time,features)
        x = self.layerNorm(x)
        # restore the original dimensions
        x = x.transpose(2,3).contiguous() # result = (batch, channel, features, time)
        x = F.gelu(x)
        x = self.dropOut(x)
        x = self.CNN1(x)
        x = x.transpose(2,3).contiguous()
        x = self.layerNorm2(x)
        x = x.transpose(2,3).contiguous()
        x = F.gelu(x)
        x = self.dropOut(x)
        x = self.CNN2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):

    def __init__(self, RNNDim, hiddenSize, dropOut, batchFirst):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU( input_size=RNNDim, hidden_size=hiddenSize, num_layers=1, batch_first=batchFirst, bidirectional=True)
        self.layerNorm = nn.LayerNorm(RNNDim)
        self.dropOut = nn.Dropout(dropOut)

    def forward(self, x):
        x = self.layerNorm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropOut(x)
        return x

class SpeechRecognition(nn.Module):
    def __init__(self, CNN_number, RNN_number, RNNCells, NoClasses, features, dropOut=0.1):
        super(SpeechRecognition, self).__init__()

        self.CNN_number = CNN_number
        self.RNN_number = RNN_number
        self.CNN = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.resCNN = ResCNN(inChannels=32,outChannels=32,kernel=3,dropOut=dropOut,features=features)

        self.fc = nn.Linear(features*32, RNNCells)
        self.BIDRNN_F = BidirectionalGRU(RNNDim=RNNCells,hiddenSize=RNNCells,dropOut=dropOut, batchFirst=True)
        self.BIDRNN   = BidirectionalGRU(RNNDim=RNNCells*2 , hiddenSize=RNNCells, dropOut=dropOut, batchFirst=False)

        self.classifier = nn.Sequential(
            nn.Linear(RNNCells*2, RNNCells),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropOut),
            nn.Linear(RNNCells, NoClasses)
        )

    def forward(self, x):
        x = self.CNN(x)
        for i in range (self.CNN_number):
            x = self.resCNN(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)                                # (batch, time, feature)
        x = self.fc(x)
        x = self.BIDRNN_F(x)
        for i in range (self.RNN_number-1):
            x = self.BIDRNN(x)
        x = self.classifier(x)
        return x
