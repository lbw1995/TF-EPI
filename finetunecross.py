from modelepi import *
from readfile import readfileforepi
from function import train_token_trans,evaluate_token_trans


def lf_fine_tuning_bertcnn_trans_train(tokenizerpath, pretrain_model_path, train_data_path, validate_data_path, modelsavepath, dataname, tokenizer_maxlen=5100, knum=6, train_batchsize=1, validate_batchsize=1, shuffle=True, learning_rate=1e-5, epochs=30, kernel_size=10, batchtimes=10):
    hidden_size = 768
    n_class = 2
    filter_sizes = [6, 6, 6]
    num_filters = 3
    expnumber = 69 #number of experimental features
    modelnew = LongformerTextcnnTrans(pretrain_model_path, num_filters, filter_sizes, n_class, hidden_size, kernel_size)
    modelnew.to(device)
    tokenizer = LongformerTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    encodings_train = readfileforepi(train_data_path, tokenizer, tokenizer_maxlen=tokenizer_maxlen, knum=knum)
    encodings_test = readfileforepi(validate_data_path, tokenizer, tokenizer_maxlen=tokenizer_maxlen, knum=knum)
    datasett = Dataset(encodings_train)
    loadert = torch.utils.data.DataLoader(datasett, batch_size=train_batchsize, shuffle=shuffle)
    
    datasetv = Dataset(encodings_test)
    loaderv = torch.utils.data.DataLoader(datasetv, batch_size=validate_batchsize, shuffle=shuffle)
    optim = AdamW(modelnew.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #criterion2 = nn.CrossEntropyLoss()
    #criterion2 = criterion2.cuda()
    for p in modelnew.parameters():
        p.requires_grad = True
    #criterion2 = nn.NLLLoss()
    #total_steps = len(datasett) * epochs
    for i in range(epochs):
        totalloss, epoch_train_loss, train_acc = train_token_trans(i, epochs, modelnew, loadert, loaderv, optim, device, criterion, modelsavepath, dataname, batchtimes)
        #train_loss, train_acc = train_token(modelnew, loadert, optim, device, criterion)
        print("total loss: ", totalloss, "\t", "train loss: ", epoch_train_loss, "\t", "train acc: ", train_acc)

def lf_fine_tuning_bertcnn_trans_validate(tokenizerpath, validate_data_path, modelsavepath, figpath, dataname, tokenizer_maxlen=5100, knum=6, validate_batchsize=1, shuffle=True, epochs=30):
    tokenizer = LongformerTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    encodings_test = readfileforepi(validate_data_path, tokenizer, tokenizer_maxlen=tokenizer_maxlen, knum=knum)
    datasetv = Dataset(encodings_test)
    criterion = nn.CrossEntropyLoss()
    loaderv = torch.utils.data.DataLoader(datasetv, batch_size=validate_batchsize, shuffle=shuffle) 
    for i in range(epochs):
        valid_loss, valid_acc = evaluate_token_trans(modelsavepath, dataname, loaderv, device, i+1, figpath, criterion)
        print("valid loss: ", valid_loss, "\t", "valid acc: ", valid_acc)               