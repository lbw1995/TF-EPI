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

def masklm(tensor):
    for i in range(tensor.shape[0]):
        seqlen = int(sum(tensor[i] > 2))
        masklen = int(seqlen * 0.15) + 1
        randstart = random.randint(1, seqlen - masklen + 2)
        #print(seqlen, masklen, randstart)
        for j in range(randstart, randstart + masklen):
            tensor[i, j] = 4
    return tensor

def plot_roc(y_test, y_score, epoch, figpath):
    fpr,tpr,threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver operating characteristic example', fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    #plt.show()
    savefigpath = figpath + "/figure_" + str(epoch) + ".png"
    plt.savefig(savefigpath)
    precision, recall, threshold = precision_recall_curve(y_test, y_score)
    aupr = auc(recall, precision)
    print("AUC:\t", roc_auc, "\nAUPRC:\t", aupr, "\n")

def getpath(dictpath, pathnum = 10):
    paths = [str(x) for x in Path(dictpath).glob('**/*')]
    if pathnum <= len(paths):
        paths0 = paths[0:pathnum]
    else:
        paths0 = paths
    return paths0

def stringtokmers(string, knum):
    kmers=''
    if len(string)<knum:
        return kmers
    for i in range(0,len(string)-knum+1):
        kmers=kmers+string[i:i+knum]+' '
    kmers=kmers[0:-1]
    return kmers    

def concatekmers(string1, string2):
    newstr=string1+'</s>'+string2
    return newstr

def train_tokens(epoch, model, dataloader, optimizer, device, criterion, scheduler, savepath, dataname):
    epoch_loss = 0
    epoch_acc = 0
    
    loop = tqdm(dataloader, leave=True)
    i = 0
    #optimizer.zero_grad()
    for batch in loop:
        i = i + len(batch['input_ids'])
        optimizer.zero_grad()
        #此处存在一定争议
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels2d = torch.zeros(labels.shape[0], 2)
        labels2d[range(labels.shape[0]), list(labels.squeeze(1).int())] = 1
        labels2d = labels2d.to(device)
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels2d)
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        
#统计training过程准确率        
        y_pred_label = outputs.argmax(dim=1)
        acc = ((y_pred_label == labels.view(-1)).sum()).item()
        epoch_acc += acc
        loop.set_postfix(loss=loss.item(), acc=epoch_acc/i)
    torch.save(model, '{0}/{1}_lftxcnnmodel_epoch_{2}.pth'.format(savepath, dataname, epoch))
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset)


def evaluate_tokens(modelpath, dataname, iterator, device, epoch, figpath, criterion):
    
    model = torch.load(os.path.join(modelpath, dataname + '_lftxcnnmodel_epoch_' + str(epoch-1) + '.pth'),map_location='cuda:0')
    
    posmatrix = torch.Tensor()
    posmatrix = posmatrix.to(device)
    
    labelmatrix = torch.Tensor()
    labelmatrix = labelmatrix.to(device)
    
    predlabel = torch.Tensor()
    predlabel = predlabel.to(device)
    
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    loop = tqdm(iterator, leave=True)
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels2d = torch.zeros(labels.shape[0], 2)
            labels2d[range(labels.shape[0]), list(labels.squeeze(1).int())] = 1
            labels2d = labels2d.to(device)
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels2d)
            y_pred_label = outputs.argmax(dim=1)
            y_pos = outputs.softmax(dim=1)
            acc = ((y_pred_label == labels.view(-1)).sum()).item()
            posmatrix = torch.cat((posmatrix,y_pos),dim=0)
            labelmatrix=torch.cat((labelmatrix,labels),dim=0)
            predlabel = torch.cat((predlabel,y_pred_label),dim=0)
            epoch_loss += loss.item()
            epoch_acc += acc
    
    y_score = posmatrix[:,1]
    y_score = y_score.to(torch.device('cpu'))
    y_test = labelmatrix.squeeze(1)
    y_test = y_test.to(torch.device('cpu'))
    
    predlabel = predlabel.to(torch.device('cpu'))
    f1w = f1_score(y_test, predlabel, average='weighted')
    f1 = f1_score(y_test, predlabel)
    print("F1w:\t", f1w, "\nF1:\t", f1)
    
    plot_roc(y_test, y_score, epoch, figpath)

    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset)

def train_token_trans(epoch, n_epoch, model, dataloaders, dataloadert, optimizer, device, criterion, savepath, dataname, batchtimes):
    total_loss = 0
    epoch_train_loss = 0
    epoch_acc = 0
    
    len_dataloader = min(len(dataloaders), len(dataloadert))
    data_source_iter = iter(dataloaders)
    data_target_iter = iter(dataloadert)
    j = 0
    loop = tqdm(range(0, len_dataloader))
    batchsize = 0
    
    optimizer.zero_grad()
    
    for i in loop:
        #if epoch==0:
        #    lamda = 1-i/len_dataloader*0.99
        #else:
        #    lamda = 0.01
        lamda = 0.01
        batch = data_source_iter.next()
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        #optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        batchsize = len(input_ids)
        j = j + len(input_ids)  
        labels2d = torch.zeros(labels.shape[0], 2)
        labels2d[range(labels.shape[0]), list(labels.squeeze(1).int())] = 1
        labels2d = labels2d.to(device)
        batch_size = len(labels2d)
        domainlabel = torch.zeros(batch_size,1).to(device)
        domainlabel = domainlabel.long()
        domainlabel2d = torch.zeros(domainlabel.shape[0], 2)
        domainlabel2d[range(domainlabel.shape[0]), list(domainlabel.squeeze(1).int())] = 1
        domainlabel2d = domainlabel2d.to(device)

        classout,domainout = model(input_ids, mask, alpha)

        classloss = criterion(classout, labels2d)
        domainloss_s = criterion(domainout,domainlabel2d)

        batch_t = data_target_iter.next()
        input_ids = batch_t['input_ids'].to(device)
        mask = batch_t['attention_mask'].to(device)
        batch_size = len(input_ids)
        targrtlabel = torch.ones(batch_size,1).to(device)
        targrtlabel = targrtlabel.long()
        targrtlabel2d = torch.zeros(targrtlabel.shape[0], 2)
        targrtlabel2d[range(targrtlabel.shape[0]), list(targrtlabel.squeeze(1).int())] = 1
        targrtlabel2d = targrtlabel2d.to(device)
        _, targetout = model(input_ids, mask, alpha)

        domainloss_t = criterion(targetout,targrtlabel2d)

        totalloss = classloss + lamda * domainloss_s + lamda * domainloss_t
        #totalloss = classloss + 0 * domainloss_s + 0 * domainloss_t
        totalloss.backward()   
        
        if i%batchtimes == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += totalloss.item()
        epoch_train_loss += classloss.item()
        
        y_pred_label = classout.argmax(dim=1)
        acc = ((y_pred_label == labels.view(-1)).sum()).item()
        epoch_acc += acc
        loop.set_postfix(totalloss=totalloss.item(), classloss=classloss.item(), acc=epoch_acc/j)
    optimizer.step()
    torch.save(model, '{0}/{1}_lftxcnnmodel_epoch_{2}.pth'.format(savepath, dataname, epoch))
    #return total_loss / len_dataloader, epoch_train_loss / len_dataloader, epoch_acc / len_dataloader / batchsize
    return total_loss / len_dataloader, epoch_train_loss / len_dataloader, epoch_acc / len_dataloader / batchsize

def evaluate_token_trans(modelpath, dataname, iterator, device, epoch, figpath, criterion):
    
    model = torch.load(os.path.join(modelpath, dataname + '_lftxcnnmodel_epoch_' + str(epoch-1) + '.pth'),map_location='cuda:0')
    
    posmatrix = torch.Tensor()
    posmatrix = posmatrix.to(device)
    
    labelmatrix = torch.Tensor()
    labelmatrix = labelmatrix.to(device)
    
    predlabel = torch.Tensor()
    predlabel = predlabel.to(device)
    
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    data_target_iter = iter(iterator)
    len_dataloader = len(iterator)
    alpha = 0
    j = 0
    loop = tqdm(range(0, len_dataloader))
    with torch.no_grad():
        for i in loop:
            batch = data_target_iter.next()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels2d = torch.zeros(labels.shape[0], 2)
            labels2d[range(labels.shape[0]), list(labels.squeeze(1).int())] = 1
            labels2d = labels2d.to(device)
            outputs, _ = model(input_ids, mask, alpha)
            loss = criterion(outputs, labels2d)
            y_pred_label = outputs.argmax(dim=1)
            y_pos = outputs.softmax(dim=1)
            acc = ((y_pred_label == labels.view(-1)).sum()).item()
            posmatrix = torch.cat((posmatrix,y_pos),dim=0)
            labelmatrix=torch.cat((labelmatrix,labels),dim=0)
            predlabel = torch.cat((predlabel,y_pred_label),dim=0)
            epoch_loss += loss.item()
            epoch_acc += acc
    
    y_score = posmatrix[:,1]
    y_score = y_score.to(torch.device('cpu'))
    y_test = labelmatrix.squeeze(1)
    y_test = y_test.to(torch.device('cpu'))
    predlabel = predlabel.to(torch.device('cpu'))
    f1w = f1_score(y_test, predlabel, average='weighted')
    f1 = f1_score(y_test, predlabel)
    print("F1w:\t", f1w, "\nF1:\t", f1)
    plot_roc(y_test, y_score, epoch, figpath)
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset)


