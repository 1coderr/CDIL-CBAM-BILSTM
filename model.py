import torch
import torch.nn as nn
from torch.nn.utils import weight_norm




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        #self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        #x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


#-----------------------------------------------------------------------------------------------------------------------#

class CDIL_Block(nn.Module):
    def __init__(self, c_in, c_out, ks, pad, dil):
        super(CDIL_Block, self).__init__()
        self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode='circular'))
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.normal_(0, 0.01)
        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        # self.lstm = nn.LSTM(input_size=1000 * 1, hidden_size=500 * 1, num_layers=6, batch_first=True,
        #                     bidirectional=True)
        # self.hidden_cell = None
        # self.drop=torch.nn.Dropout(p=0.5,inplace=False)
        #self.eca = eca_layer(64)
        self.cbam=cbam_block(2)
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        self.nonlinear = nn.ReLU()
        self.BN=nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        y_conv1 = self.conv(x)
        # y_conv2, (h1, b1) = self.lstm(y_conv1, self.hidden_cell)
        # y_conv = y_conv1 + y_conv2
        #y_conv = torch.cat((y_conv1, y_conv2),0)
        # y_conv=self.drop(y_conv)
        #out = self.eca(y_conv1)
        out=self.cbam(y_conv1)
        res = x if self.res is None else self.res(x)
        out=self.nonlinear(y_conv1)
        return self.BN(out) + res
        #return self.nonlinear(y_conv1)+res

class CDIL_ConvPart(nn.Module):
    def __init__(self, dim_in, hidden_channels, ks=3):
        super(CDIL_ConvPart, self).__init__()
        layers = []
        num_layer = len(hidden_channels)#---------------------------[hidden]
        for i in range(num_layer):
            this_in = dim_in if i == 0 else hidden_channels[i - 1]
            this_out = hidden_channels[i]#--------------------[hidden]
            this_dilation = 2 ** i
            this_padding = int(this_dilation * (ks - 1) / 2)
            layers += [CDIL_Block(this_in, this_out, ks, this_padding, this_dilation)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)

class CDIL_CNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, use_embed=False, char_vocab=None):
        super(CDIL_CNN, self).__init__()

        self.use_embed = use_embed
        if self.use_embed:
            self.embedding = nn.Embedding(char_vocab, input_size)

        self.conv = CDIL_ConvPart(input_size, num_channels, kernel_size)
        self.Bilstm = nn.LSTM(input_size=2500 * 1, hidden_size=1250 * 1, num_layers=6,batch_first=True, bidirectional=True)
        self.hidden_cell=None
        #self.drop = torch.nn.Dropout(p=0.5, inplace=False)
        self.classifier = nn.Linear(32, output_size)#-------------------------num_channels[-1]
        self.SVM = SVMClassifier(6)
        #self.softmax=nn.Softmax(dim=0)
    def forward(self, x):
        if self.use_embed:
            x = self.embedding(x)
            x = x.permute(0, 2, 1).to(dtype=torch.float)
        y_conv1 = self.conv(x)  # x, y: num, channel(dim), length
        y_conv2,(h1,b1)=self.Bilstm(y_conv1,self.hidden_cell)
        y_conv=y_conv1+y_conv2
        #y_conv=torch.cat((y_conv1,y_conv2),1)
        #y_conv = self.drop(y_conv)
        y = self.classifier(torch.mean(y_conv, dim=2))
        y=self.SVM(y)
        #y=self.softmax(y)
        return y



SEQ_LEN = 10000
import numpy as np
LAYER = int(np.log2(SEQ_LEN))
model=CDIL_CNN(input_size=1, output_size=6, num_channels=[32]*LAYER)
#print(model)