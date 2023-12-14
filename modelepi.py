#-*- coding: utf-8 -*-
from tqdm.auto import tqdm
import os
import numpy as np
import random
import sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import matplotlib
from matplotlib import pyplot as plt

import torch
import torchvision
from transformers import RobertaTokenizer, LongformerTokenizer, AdamW, LongformerForSequenceClassification, LongformerForMaskedLM, LongformerConfig, RobertaModel, RobertaForSequenceClassification, LongformerModel, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

from torch.nn import init
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Function

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings['labels'].shape[0]
        #return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        return {key: tensor[i] for key, tensor in self.encodings.items()}
        
class TextCNN(nn.Module):
    def __init__(self,num_filters, filter_sizes, n_class, hidden_size, kernel_size):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)
        t = 5095 // kernel_size
        self.Weight = nn.Linear(self.num_filter_total * t, n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([n_class]))
        self.filter_list = nn.ModuleList([
          nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])
        self.kernel_size = kernel_size
        self.t = t
        self.filter_sizes = filter_sizes

    def forward(self, x):
        x = x.unsqueeze(1) # [bs, channel=1, seq, hidden]    
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x)) # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size = (self.kernel_size, 1) #kernel size set????????????????????????????????
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)
        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total * self.t])
        output = self.Weight(h_pool_flat) + self.bias # [bs, n_class] ??? why use bias 如果不用行不行
        #output = self.Weight(h_pool_flat) # [bs, n_class]
        return output
        
class RobertTextcnn(nn.Module):
    def __init__(self, pretrain_path, num_filters, filter_sizes, n_class, hidden_size, kernel_size):
        super(RobertTextcnn,self).__init__()
        self.bert=RobertaModel.from_pretrained(pretrain_path)  #从路径加载预训练模型
        self.cnn = TextCNN(num_filters, filter_sizes, n_class, hidden_size, kernel_size)
        

    def forward(self,input_ids,mask):
        output = self.bert(input_ids,mask)
        linear_in = torch.reshape(output[0],(output[0].shape[0],-1))
        #linear_in = torch.cat((linear_in,expdata),dim=1)
        linear_in = linear_in.reshape([-1,5100,768])
        out=self.cnn(linear_in)
        return out             

class LongformerTextcnnTrans(nn.Module):
    def __init__(self, pretrain_path, num_filters, filter_sizes, n_class, hidden_size, kernel_size):
        super(LongformerTextcnnTrans,self).__init__()
        self.bert=LongformerModel.from_pretrained(pretrain_path)  #从路径加载预训练模型
        self.classifycnn = TextCNN(num_filters, filter_sizes, n_class, hidden_size, kernel_size)
        self.domaincnn = TextCNN(num_filters, filter_sizes, n_class, hidden_size, kernel_size)
        #self.GRL=GRL()
        
    def forward(self,input_ids,mask,alpha): 
        output = self.bert(input_ids,mask)
        linear_in = torch.reshape(output[0],(output[0].shape[0],-1))
        linear_in = linear_in.reshape([-1,5100,768])
        classout=self.classifycnn(linear_in)
        linear_in = GRL.apply(linear_in,alpha)
        domainout = self.domaincnn(linear_in)
        return classout,domainout