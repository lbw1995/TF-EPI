from tqdm.auto import tqdm
import torch
import torchvision
from transformers import RobertaTokenizer, LongformerTokenizer, AdamW, LongformerForSequenceClassification, LongformerForMaskedLM, LongformerConfig, RobertaModel, RobertaForSequenceClassification, LongformerModel, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

from torch.nn import init
from torch import nn, optim
import torch.nn.functional as F

from modelepi import *
from pathlib import Path
from function import getpath,stringtokmers,masklm

def lf_pre_training(tokenizerpath, inputpaths, model_output_path, tokenizer_maxlen=5100, batchsize=1, shuffle=True, attention_window=100, hidden_size=768, num_attention_heads=12, num_hidden_layers=4, type_vocab_size=1, learning_rate=1e-5, epochs=3):
    if not Path(tokenizerpath).is_dir():
        print("Error! You must input a correct tokenizer dir path.")
        return
    
    if not Path(inputpaths).is_dir():
        print("Error! You must input a correct input files path.")
        return
    
    tokenizer = LongformerTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    input_ids = []
    mask = []
    labels = []
    
    paths0 = getpath(inputpaths, pathnum=1000)
    
    for path in tqdm(paths0):
        newline = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]
        for line in lines:
            newline.append(stringtokmers(line, 6))
        #sample = tokenizer(lines, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        sample = tokenizer(newline, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        labels.append(sample.input_ids)
        mask.append(sample.attention_mask)
        input_ids.append(masklm(sample.input_ids.detach().clone()))
    input_ids=torch.cat(input_ids)
    mask=torch.cat(mask)
    labels=torch.cat(labels)

    encodings = {
        'input_ids' : input_ids,
        'attention_mask' : mask,
        'labels' : labels
    }
    
    dataset = Dataset(encodings)
    torch.cuda.empty_cache()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    attention_window2 = [100,100,100,5100]
    config = LongformerConfig(
        attention_window=attention_window2,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=tokenizer_maxlen+2,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size
    )
    model = LongformerForMaskedLM(config)
    model.to(device)
    optim2 = AdamW(model.parameters(), lr=learning_rate)
    epochs = epochs
    
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        epoch_loss = 0
        for batch in loop:
            #optim.zero_grad()
            optim2.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=mask,
                                labels=labels)
            loss = outputs.loss
            loss.backward()
            #optim.step()
            optim2.step()
            epoch_loss += loss.item()

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss=loss.item())
        #print("epoch loss: ", epoch_loss / len(dataloader))
    model.save_pretrained(model_output_path+"longformermodel")
    print("Pretain over!")

def robert_lf_pre_training(tokenizerpath, inputpaths, model_output_path, tokenizer_maxlen=5100, batchsize=1, shuffle=True, hidden_size=768, num_attention_heads=12, num_hidden_layers=4, type_vocab_size=1, learning_rate=1e-5, epochs=3):
    if not Path(tokenizerpath).is_dir():
        print("Error! You must input a correct tokenizer dir path.")
        return
    
    if not Path(inputpaths).is_dir():
        print("Error! You must input a correct input files path.")
        return
    
    tokenizer = RobertaTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    input_ids = []
    mask = []
    labels = []
    
    paths0 = getpath(inputpaths, pathnum=1000)
    
    for path in tqdm(paths0):
        newline = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]
        for line in lines:
            newline.append(stringtokmers(line, 6))
        #sample = tokenizer(lines, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        sample = tokenizer(newline, max_length=tokenizer_maxlen, padding='max_length', truncation=True, return_tensors='pt')
        labels.append(sample.input_ids)
        mask.append(sample.attention_mask)
        input_ids.append(masklm(sample.input_ids.detach().clone()))
    input_ids=torch.cat(input_ids)
    mask=torch.cat(mask)
    labels=torch.cat(labels)

    encodings = {
        'input_ids' : input_ids,
        'attention_mask' : mask,
        'labels' : labels
    }
    
    dataset = Dataset(encodings)
    torch.cuda.empty_cache()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=tokenizer_maxlen+2,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size
    )
    model = RobertaForMaskedLM(config)
    model.to(device)
    optim2 = AdamW(model.parameters(), lr=learning_rate)
    epochs = epochs
    
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        epoch_loss = 0
        for batch in loop:
            #optim.zero_grad()
            optim2.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=mask,
                                labels=labels)
            loss = outputs.loss
            loss.backward()
            #optim.step()
            optim2.step()
            epoch_loss += loss.item()

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss=loss.item())
        #print("epoch loss: ", epoch_loss / len(dataloader))
    model.save_pretrained(model_output_path+"longformermodel")
    print("Pretain over!")
