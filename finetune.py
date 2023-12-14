from modelepi import *
from readfile import readfileforepi
from function import train_tokens, evaluate_tokens


def lf_fine_tuning_bertcnn_train(modelname = "Robert", tokenizerpath, pretrain_model_path, train_data_path, modelsavepath, dataname, tokenizer_maxlen=5100, knum=6, train_batchsize=2, shuffle=True, learning_rate=1e-5, epochs=30, kernel_size=10):
    hidden_size = 768
    n_class = 2
    filter_sizes = [6, 6, 6]
    num_filters = 3
    expnumber = 69 #number of experimental features
    if modelname=="Longformer":
        modelnew = LongformerTextcnn(pretrain_model_path, num_filters, filter_sizes, n_class, hidden_size, kernel_size)
        modelnew.to(device)
        tokenizer = LongformerTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    elif modelname=="Robert":
        modelnew = RobertTextcnn(pretrain_model_path, num_filters, filter_sizes, n_class, hidden_size, kernel_size)
        modelnew.to(device)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    else:
        print("This is not a effective model name!")
        return
    encodings_train = readfileforepi(train_data_path, tokenizer, tokenizer_maxlen=tokenizer_maxlen, knum=knum)
    datasett = Dataset(encodings_train)
    loadert = torch.utils.data.DataLoader(datasett, batch_size=train_batchsize, shuffle=shuffle)
    optim = AdamW(modelnew.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(datasett) * epochs
    warm_up_ratio = 0.05
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    #epochs = epochs
    for i in range(epochs):
        train_loss, train_acc = train_tokens(i, modelnew, loadert, optim, device, criterion, scheduler, modelsavepath, dataname)
        #train_loss, train_acc = train_token(modelnew, loadert, optim, device, criterion)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)
        
def lf_fine_tuning_bertcnn_validate(tokenizerpath, validate_data_path, modelsavepath, figpath, dataname, tokenizer_maxlen=5100, knum=6, validate_batchsize=1, shuffle=True, epochs=30, kernel_size=10):
    tokenizer = LongformerTokenizer.from_pretrained(tokenizerpath, max_len=tokenizer_maxlen)
    encodings_test = readfileforepi(validate_data_path, tokenizer, tokenizer_maxlen=tokenizer_maxlen, knum=knum)
    datasetv = Dataset(encodings_test)
    criterion = nn.CrossEntropyLoss()
    loaderv = torch.utils.data.DataLoader(datasetv, batch_size=validate_batchsize, shuffle=shuffle)
    for i in range(epochs):
        valid_loss, valid_acc = evaluate_tokens(modelsavepath, dataname, loaderv, device, i+1, figpath, criterion)
        print("valid loss: ", valid_loss, "\t", "valid acc: ", valid_acc) 