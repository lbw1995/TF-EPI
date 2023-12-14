import torch
from pathlib import Path
from transformers import LongformerTokenizer, RobertaTokenizer
from function import stringtokmers, concatekmers


def readfileforepi(filepath, tokenizer, tokenizer_maxlen=5100, knum=6):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[1:-1]
    newline=[]
    labellines=[]
    count = 0

    input_ids = []
    mask = []
    labels = []

    #expvalues = torch.tensor([])

    for line in lines:
        newline=[]
        labellines=[]
        if count%100==0:
            print("nuber of data:", count)
        elements = line.split(",")
        newline.append(concatekmers(stringtokmers(elements[2], knum), stringtokmers(elements[3], knum)))
        labellines.append(elements[4])
        #for i in range(5,5+expnumber):
        #    expvalues = torch.cat((expvalues,torch.tensor([float(elements[i])])))
        count = count + 1
        sample = tokenizer(newline, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(sample.input_ids)
        mask.append(sample.attention_mask)
        labellines = list(map(int, labellines))
        label = torch.tensor(labellines)
        label = label.unsqueeze(dim=0).t()
        labels.append(label)
    input_ids=torch.cat(input_ids)
    mask=torch.cat(mask)
    labels=torch.cat(labels)
    #expvalues = expvalues.reshape(-1,expnumber)
    encodings = {
        'input_ids' : input_ids,
        'attention_mask' : mask,
        'labels' : labels
    }
    return encodings

def readfileforepi5(filepath, tokenizer, tokenizer_maxlen=5100, knum=6):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[1:-1]
    newline=[]
    labellines=[]
    count = 0

    input_ids = []
    mask = []
    labels = []
    promoterseqs = []

    #expvalues = torch.tensor([])

    for line in lines:
        newline=[]
        labellines=[]
        if count%100==0:
            print("nuber of data:", count)
        elements = line.split(",")
        newline.append(concatekmers(stringtokmers(elements[2], knum), stringtokmers(elements[3], knum)))
        labellines.append(elements[4])
        promoterseqs.append(elements[3])
        #for i in range(5,5+expnumber):
        #    expvalues = torch.cat((expvalues,torch.tensor([float(elements[i])])))
        count = count + 1
        sample = tokenizer(newline, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(sample.input_ids)
        mask.append(sample.attention_mask)
        labellines = list(map(int, labellines))
        label = torch.tensor(labellines)
        label = label.unsqueeze(dim=0).t()
        labels.append(label)
    input_ids=torch.cat(input_ids)
    mask=torch.cat(mask)
    labels=torch.cat(labels)
    #expvalues = expvalues.reshape(-1,expnumber)
    encodings = {
        'input_ids' : input_ids,
        'attention_mask' : mask,
        'labels' : labels,
        'promoterseqs' : promoterseqs
    }
    return encodings

def readfileforepi6(filepath, tokenizer, tokenizer_maxlen=5100, knum=6):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[1:-1]
    newline=[]
    labellines=[]
    count = 0

    input_ids = []
    mask = []
    labels = []
    enhancerseqs = []

    #expvalues = torch.tensor([])

    for line in lines:
        newline=[]
        labellines=[]
        if count%100==0:
            print("nuber of data:", count)
        elements = line.split(",")
        newline.append(concatekmers(stringtokmers(elements[2], knum), stringtokmers(elements[3], knum)))
        labellines.append(elements[4])
        enhancerseqs.append(elements[2])
        #for i in range(5,5+expnumber):
        #    expvalues = torch.cat((expvalues,torch.tensor([float(elements[i])])))
        count = count + 1
        sample = tokenizer(newline, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(sample.input_ids)
        mask.append(sample.attention_mask)
        labellines = list(map(int, labellines))
        label = torch.tensor(labellines)
        label = label.unsqueeze(dim=0).t()
        labels.append(label)
    input_ids=torch.cat(input_ids)
    mask=torch.cat(mask)
    labels=torch.cat(labels)
    #expvalues = expvalues.reshape(-1,expnumber)
    encodings = {
        'input_ids' : input_ids,
        'attention_mask' : mask,
        'labels' : labels,
        'enhancerseqs' : enhancerseqs
    }
    return encodings

def readfileforepi7(filepath, tokenizer, tokenizer_maxlen=5100, knum=6):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[1:-1]
    newline=[]
    labellines=[]
    count = 0

    input_ids = []
    mask = []
    labels = []
    enhancerseqs = []
    promoterseqs = []
    #expvalues = torch.tensor([])

    for line in lines:
        newline=[]
        labellines=[]
        if count%100==0:
            print("nuber of data:", count)
        elements = line.split(",")
        newline.append(concatekmers(stringtokmers(elements[2], knum), stringtokmers(elements[3], knum)))
        labellines.append(elements[4])
        promoterseqs.append(elements[3])
        enhancerseqs.append(elements[2])
        #for i in range(5,5+expnumber):
        #    expvalues = torch.cat((expvalues,torch.tensor([float(elements[i])])))
        count = count + 1
        sample = tokenizer(newline, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(sample.input_ids)
        mask.append(sample.attention_mask)
        labellines = list(map(int, labellines))
        label = torch.tensor(labellines)
        label = label.unsqueeze(dim=0).t()
        labels.append(label)
    input_ids=torch.cat(input_ids)
    mask=torch.cat(mask)
    labels=torch.cat(labels)
    #expvalues = expvalues.reshape(-1,expnumber)
    encodings = {
        'input_ids' : input_ids,
        'attention_mask' : mask,
        'labels' : labels,
        'enhancerseqs' : enhancerseqs,
        'promoterseqs' : promoterseqs
    }
    return encodings

def load_tsv_of_promoter(neg_data_path): #load promoterseqs
    neg_sequences = []
    with open(neg_data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[1:-1]
    for line in lines:
        elements = line.split(",")
        neg_sequences.append(elements[3])
    return neg_sequences

def load_tsv_of_enhancer(neg_data_path): #load enhancerseqs
    neg_sequences = []
    with open(neg_data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[1:-1]
    for line in lines:
        elements = line.split(",")
        neg_sequences.append(elements[2])
    return neg_sequences