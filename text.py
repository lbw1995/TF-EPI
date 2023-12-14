

def text_for_print(option):
    if option == 1:
        print('''
Welcome to use TF-EPI!

You are able to choose many type of functions.

Pretrain the model-----------Use multiple sequences to get a pretrained model.

Fine-tune the model-----------Use your pretrained model to get a classifier model of EPI.

De-novo discovery of motifs-----------Use attention matrix of Transformer to discover biological meaningful motifs.

Discovery of interaction of k-mers-----------Use attention matrix of Transformer to discover k-mers with interactions
        ''')
    elif option == 2:
        print('''
This is the used for pretraining the model.

Parameters required:

Tokenizer path: the path of tokenizer.

Input dir path: the path of data dir for pretraining.

Model save path: the save path of the pretrained model.

Batchsize: batch size of the pretraining steps. Default:1

Number of hidden layers: the number of Transformer encoders. Default:4

Number of tokenizer: the number of max tokenizer length. Default:5100                
            ''')
    elif option == 3:
        print('''
This is the used for pretraining the model.

Parameters required:

Tokenizer path: the path of tokenizer.

Input dir path: the path of data dir for pretraining.

Model save path: the save path of the pretrained model.

Batchsize: batch size of the pretraining steps. Default:2

Number of hidden layers: the number of Transformer encoders. Default:4

Length of tokenizer: the number of max tokenizer length. Default:5100                
                ''')
    elif option == 4:
        print('''
This is the used for Finetuning the model on the training set.

Parameters required:

Name of model: use Roberta model or Longformer model

Tokenizer path: the path of tokenizer.

Input train data path: the path of training dataset.

Pretrining model path: the save path of the pretrained model.

Data name: the name of dataset.

Length of tokenizer: the number of max tokenizer length. Default:5100

Model save path: the save path of the fine-tuned model.

Batchsize: batch size of the training steps. Default:2

Epoch : Default 30
                ''')
    elif option == 5:
        print('''
This is the used for Validating the model on the validation set.

Parameters required:

Tokenizer path: the path of tokenizer.

Input validate data path: the path of validation dataset.

Pretrining model path: the save path of the pretrained model.

Model save path: the save path of the fine-tuned model.

Figure path: the save dir path of the validation ROC plot

Data name: the name of dataset.

Batchsize: batch size of the validation steps. Default:1

Epoch : should be same as the number of Epoch in trainng steps. Default 30
        ''')
    elif option == 6:
        print('''
This is the used for Finetuning the model on the cross cell line training set.

Parameters required:

Tokenizer path: the path of tokenizer.

Input train data path: the path of training dataset.

Input validate data path: the path of validation dataset.

Pretrining model path: the save path of the pretrained model.

Data name: the name of dataset.

Model save path: the save path of the fine-tuned model.

Training batchsize: batch size of the training dataset. Default:1

Validating batchsize: batch size of the validation dataset. Default:1

Epoch : Default 30

        ''')
    elif option == 7:
        print('''
This is the used for validate the fine-tuned model on the cross cell line training set.

Parameters required:

Tokenizer path: the path of tokenizer.

Input validate data path: the path of validation dataset.

Model save path: the save path of the fine-tuned model.

Figure path: the save dir path of the validation ROC plot

Data name: the name of dataset.

Validating batchsize: batch size of the validation dataset. Default:1

Epoch : Default 30
        ''')
    elif option == 8:
        print('''
This is used for the discovery of motifs.

Parameters required:

Name of model: use Roberta model or Longformer model

Region of motifs: Enhance region or Promoter region

Tokenizer path: the path of tokenizer.

Input possitive data path: dataset of possitive samples.

Input negative data path: dataset of negative samples.

Model path: the path of trained model.

Motif save path: the save path of the motifs.
        ''')
    elif option == 9:
        print('''
This is used for the discovery of k-mer-k-mer interactions.

Parameters required:

Name of model: use Roberta model or Longformer model

Region of motifs: Enhance region or Promoter region

Tokenizer path: the path of tokenizer.

Input possitive data path: dataset of possitive samples.

Interaction save path: the save path of the k-mer-k-mer interactions.

P-value: cut off value of the high attention k-mer-k-mer interactions.

Cut number: number of output top k-mer-k-mer with high interactions.
        ''')
    