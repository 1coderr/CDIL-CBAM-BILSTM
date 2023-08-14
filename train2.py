import itertools

import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision
import math
import random
from PIL import Image,ImageOps,ImageEnhance
import numbers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pprint import pprint
import sys
from matplotlib import pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
#from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
# from torchvision.ops import FrozenBatchNorm2d
#from torchvision.ops import FrozenBatchNorm2d
from transformers import AdamW
import time
from datetime import datetime
from config import epochs, batch_size, number2name,LR
from data_helper import MyDataset, list2tensor


from utils import writelog, save_checkpoint_state, standardize
import matplotlib.pyplot as plt
from model import CDIL_CNN,One_CNN,CNN_CBAM_BiLSTM,OneDimConvDenseNet,AF_CNN
#from model2 import Learn,CIFAR10Model,SCU
#from model3 import AF_CNN,Model,OneDimConvDenseNet
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import wandb
from net_RF import receptive_field

# wandb.init(project="CDIL_CNN", entity="zhinengganzhi")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 0.01,
#   "batch_size": 32
# }


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
abnormal_arr = np.load(r'data/DAS_10k_arr.npy')

# 张量标准化
abnormal_arr, mean_value, deviation = standardize(abnormal_arr)
abnormal_label = np.load(r'data/DAS_10k_label.npy')

arr_train, arr_val, labels_train, labels_val = train_test_split(abnormal_arr, abnormal_label,
                                                                test_size=0.3,
                                                                shuffle=True)
train_set = MyDataset(arr_train, labels_train)
val_set = MyDataset(arr_val, labels_val)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
#--------------------------------------------------训练模型---------------------------------------
#模型实例化
a=[]
b=[]
# model=SCU(num_classes=2)chu
SEQ_LEN = 10000
LAYER = int(np.log2(SEQ_LEN))
#print([SEQ_LEN]*LAYER)
#model=CDIL_CNN(input_size=1, output_size=6, num_channels=[32]*LAYER)
model=One_CNN()
print(model)

#定义优化器
optimizer=optim.Adam(model.parameters(),lr=LR)
#定义损失函数
criterion=nn.CrossEntropyLoss(reduction='sum')
exp_lr_schedular=lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
if torch.cuda.is_available():
    model=model.cuda()
    criterion=criterion.cuda()

import logging
file_name = '10k_ceshi'
os.makedirs('time_log', exist_ok=True)
os.makedirs('time_model', exist_ok=True)
log_file_name = './time_log/' + file_name + '.txt'
model_name = './time_model/' + file_name + '.pt'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

now = time.localtime()
tm = time.strftime("%Y%m%d%H%M%S", now)
log_file = f'log/train_{tm}_log.txt'
val_epoch_min_loss = sys.maxsize
max_map = 0
para_num = sum(p.numel() for p in model.parameters() if p.requires_grad)#参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loginf(torch.cuda.get_device_name(device))
SEQ_LEN = len(arr_train[1])
LAYER = int(np.log2(SEQ_LEN))
receptive_field(seq_length=SEQ_LEN, kernel_size=3, layer=LAYER)
loginf("model参数数量: {}".format(para_num))
from tqdm import tqdm
#------------------------------------------------训练网络------------------------------------------
def train(epoch):
    train_loss=0
    model.train()
    t_start = datetime.now()
    #更新学习率
    exp_lr_schedular.step()
    #从train_loader读取数据，每一次迭代data都是一个batch的数据
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=Variable(data),Variable(target)
        if torch.cuda.is_available():
            data=data.cuda()
            target=target.cuda()
        optimizer.zero_grad()
        output=model(data)
        Loss=criterion(output,target.long().cuda())
        Loss.backward()
        optimizer.step()
        model.zero_grad()  # 梯度清零
        train_loss += Loss.item()
        wandb.log({"loss": Loss})

        # Optional
        wandb.watch(model)
        if(batch_idx+1)%5==0:
            loginf('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch,(batch_idx+1)*len(data),len(train_loader.dataset),
                100.*(batch_idx+1)/len(train_loader),Loss.item()))
    t_end = datetime.now()
    epoch_time = (t_end - t_start).total_seconds()
    train_loss_mean = train_loss / len(train_loader)
    loginf('Train num: {} — Train loss: {:.6f} — Time: {}'.format(len(train_loader.dataset), train_loss_mean, epoch_time))
    loginf('_' * 80)


#定义评估函数
def evaluate(data_loader):
    model.eval()
    Loss=0
    correct=0
    t_start = datetime.now()
    targets=[]
    preds=[]
    for data,target in data_loader:
        data,target=Variable(data,volatile=True),Variable(target)
        if torch.cuda.is_available():
            data=data.cuda()
            target=target.cuda()
        output=model(data)
        Loss+=F.cross_entropy(output,target.long(),size_average=False).item()
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
        #print(target.cpu())
        #print(pred.cpu())
        targets.extend(target.cpu())
        preds.extend([p[0] for p in pred.cpu()])
    Loss/=len(data_loader.dataset)
    a.append(Loss)
    b.append(100.*correct/len(data_loader.dataset))
    #print(targets)
    #print(preds)
    p = precision_score(targets, preds, average='macro')
    r = recall_score(targets, preds, average='macro')
    f1score = f1_score(targets, preds, average='macro')
    loginf('\nval loss:{:.4f},Accuracy:{}/{}({:.3f}%)'.format(
        Loss,correct,len(data_loader.dataset),
        100.*correct/len(data_loader.dataset)))
    #hunxiao = confusion_matrix(target.cpu(), pred.cpu())
    # precision = precision_score(target.cpu(), pred.cpu())
    # recall = recall_score(target.cpu(), pred.cpu())
    # f1 = f1_score(target.cpu(), pred.cpu())

    loginf("精确率R: {:.3f}".format(p))
    loginf("召回率:{:.3f}".format(r))
    loginf("F1值:{:.3f}".format(f1score))
    t_end = datetime.now()
    epoch_time = (t_end - t_start).total_seconds()
    loginf('Time: {}'.format(epoch_time))
    loginf('_' * 120)
    saving_best=0
    if 100.*correct/len(data_loader.dataset) >= saving_best:
        saving_best = 100.*correct/len(data_loader.dataset)
        torch.save(model.state_dict(), model_name)


#绘制混淆矩阵
#----------------------------------------------

def plot_confuse(data_loader):
    targets = []
    preds = []
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        targets.extend(target.cpu())
        preds.extend([p[0] for p in pred.cpu()])

    cm = confusion_matrix(y_true=targets, y_pred=preds,labels=[0, 1, 2, 3, 4, 5])
    plt.figure()
    # 指定分类类别
    classes = [0, 1, 2, 3, 4, 5]
    title='Confusion matrix'
   #混淆矩阵颜色风格
    cmap=plt.cm.jet
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
   # 按照行和列填写百分比数据
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('10k_ceshi.jpg', dpi=300)
n_epochs=epochs
for epoch in range(n_epochs):
    train(epoch)
    evaluate(val_loader)
plot_confuse(val_loader)
# torch.save(model.state_dict(),'1D-CNN-model.ckpt')#保存为model.ckpt


import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
#1.训练时先新建个列表，然后将loss值调用列表的append方法存入列表中
#2.例如列表train_recon_loss，Discriminator_loss...，然后将列表名替换train_recon_loss，Discriminator，利用plot即可画出曲线
#3.最后将画的图保存成图片，imgpath为自定义的图片保存路径。
# plt.figure(num = 2, figsize=(640,480))
plt.figure()
plt.plot(a,'b',label = 'Recon_loss')
plt.ylabel('Recon_loss')
plt.xlabel('iter_num')
plt.legend()
plt.savefig(os.path.join(r'D:\Desktops\CDIL-CBAM-BiLSTM\figures',"10k_ceshi.jpg"))

plt.switch_backend('Agg')
#1.训练时先新建个列表，然后将loss值调用列表的append方法存入列表中
#2.例如列表train_recon_loss，Discriminator_loss...，然后将列表名替换train_recon_loss，Discriminator，利用plot即可画出曲线
#3.最后将画的图保存成图片，imgpath为自定义的图片保存路径。
# plt.figure(num = 2, figsize=(640,480))
plt.figure()
plt.plot(b,'b',label = 'Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('iter_num')
plt.legend()
plt.savefig(os.path.join(r'D:\Desktops\CDIL-CBAM-BiLSTM\figures',"10k_ceshi.jpg"))


