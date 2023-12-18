# TF-EPI
TF-EPI: An Interpretable Enhancer Promoter Interaction Detection Method Based on Large Language Model

# What is TF-EPI?
The detection of enhancer-promoter interactions (EPI) is crucial for understanding gene expression regulation, disease mechanisms, and more. In this study, we developed TF-EPI, a deep learning model based on transformer technology, designed to detect these interactions solely from DNA sequences. 

# Install
TF-EPI is a Python-based software. It is recommended to run the program using Python 3.8. The required dependency packages include:

    pytorch 1.10.0
    numpy
    pandas
    tqdm
    matplotlib
    scikit-learn
    transformers 4.29.2
    os
    sys
    ahocorasick
    operator
    scipy
    statsmodels
    biopython 1.79

## Install from source
Download the compressed source file TF-EPI.tar.gz and do as follows:

    $ tar -xzvf TF-EPI.tar.gz
    $ cd ./TF-EPI
    $ python ./main.py [options]
## Required parameters:
TF-EPI comprises various program modules, including pretraining, cell type-specific fine-tuning, cross cell type fine-tuning, motif discovery, and interact k-mers discovery.

To utilize the different modules, it is necessary to specify the required module after 'python main.py' and provide the necessary parameters：

    python ./main.py [module name] [parameters]
### Pretraining
This module is used to pretrain a model with your own DNA sequences. To use this module, you first need to prepare some DNA sequences and save them in a folder. TF-EPI will read your folder and train the pretrain model based on all the DNA sequence files in your folder. A demo example is provided in the data folder to show the files required for pretraining.

For usage:

    python ./main.py rbpretraining [tokenizer path] [input dir path] [model save path] [batch size] [number hidden layers] [max tokenizer length]
Parameters required:

Tokenizer path : the path of tokenizer. You are suggested to use the tokenizer under the folder tokenizer. 

The input dir path : the folder path of DNA sequences.

The model save path : the path you used to save the pretrained model.

The batch size : Default 2

Number hidden layers : Number of transformer encoders of your pretrained model. Default 4

Max tokenizer length : Max tokens of your input sequences. Default: 5100

If you do not want to pretrain a model by yourself, we prepare a pretrained model under the folder pretrained_model.

### Cell type-specific fine-tuning
This module is used to fine-tune a model with your own enhancer promoter pairs. To use this module, you should input split your dataset into training set and validation set and seperately do the training and validation. The data should be a csv format file with the following columns:

    enhancer_name,promoter_name,enhancer_seq,promoter_seq,label
We give an example of the input file and you can generater your own input file with the same format.
For training process, you should do:

    python ./main.py finetuningbertcnntrain [modelname] [tokenizer path] [pretraining model path] [training data path] [model save path] [data mane] [batch size] [epoch]
Parameters required:

Modelname : the type of the model, only avaliable for "Robert" and "Longformer". You are recommended to use "Robert".

Tokenizer path: the path of tokenizer.

Input train data path: the path of training dataset.

Pretrining model path: the save path of the pretrained model.

Model save path: the save folder of the fine-tuned model. The model of each epoch will be saved in this folder and used for the following validation.

Data name: the name of dataset. Used to seperate different cell line.

Batchsize: batch size of the training steps. Default:2

Epoch : Default 30

For the validation process, you should do:

     python ./main.py finetuningbertcnnvalidate [tokenizer path] [validation data path] [model save path] [figure path] [data name] [batch size] [epoch]
This is the used for Validating the model on the validation set.

Parameters required:

Tokenizer path: the path of tokenizer.

Input validate data path: the path of validation dataset.

Pretrining model path: the save folder path of the pretrained model.

Model save path: the save path of the fine-tuned model.

Figure path: the save dir path of the validation ROC plot

Data name: the name of dataset.

Batchsize: batch size of the validation steps. Default:1

Epoch : should be same as the number of Epoch in trainng steps. Default 30

### Cross cell type fine-tuning
This module is used to fine-tune a model with cross cell type dataset. You can fine-tune your model on dataset of one cell type and validate your model on another cell type. The data format is same as the cell type-specific fine-tuning dataset. You should prepare two different cell type dataset to train your model.

For training process, you should do:

    python ./main.py finetuningbertcnntranstrain [tokenizer path] [pretraining model path] [training data path] [validation data path] [model save path] [data mane] [batch size] [kernel size] [epoch]
Parameters required:

Tokenizer path: the path of tokenizer.

Pretrining model path: the save path of the pretrained model.

Input train data path: the path of training dataset.

Input validate data path: the path of validation dataset.

Data name: the name of dataset.

Model save path: the save path of the fine-tuned model.

Training batchsize: batch size of the training dataset. Default:1

Validating batchsize: batch size of the validation dataset. Default:1

Epoch : Default 30

For the validation process, you should do:

    python ./main.py finetuningbertcnntransvalidate [tokenizer path] [validation data path] [model save path] [figure path] [data mane] [batch size] [epoch]
Parameters required:

Tokenizer path: the path of tokenizer.

Input validate data path: the path of validation dataset.

Model save path: the save path of the fine-tuned model.

Figure path: the save dir path of the validation ROC plot

Data name: the name of dataset.

Validating batchsize: batch size of the validation dataset. Default:1

Epoch : should be same as the number of Epoch in trainng steps. Default 30

### Motif discovery
This module is used for the discovery of motifs based on the attention matix of transformer encoders.

For usage:

    python ./main.py motifdiscovery [model name] [region of motifs] [tokenizer path] [input possitive data path] [input negative data path] [model path] [motif save path]
Parameters required:

Name of model: use Roberta model or Longformer model

Region of motifs: Enhance region or Promoter region

Tokenizer path: the path of tokenizer.

Input possitive data path: dataset of possitive samples.

Input negative data path: dataset of negative samples.

Model path: the path of trained model.

Motif save path: the save path of the motifs.

### Interact k-mers discovery
This module is used for the discovery of high frequency k-mer-k-mer interactions.

For usage:

     python ./main.py interactkmerdiscovery [model name] [region of motifs] [input data path] [model path] [interaction save path] [p value] [cut number]
Parameters required:

Name of model: use Roberta model or Longformer model

Region of motifs: Enhance region or Promoter region

Tokenizer path: the path of tokenizer.

Input possitive data path: dataset of possitive samples.

Interaction save path: the save path of the k-mer-k-mer interactions.

P-value: cut off value of the high attention k-mer-k-mer interactions.

Cut number: number of output top k-mer-k-mer with high interactions.

## Demo test
To help you use our method, we have uploaded a demo file for you to test, and provide the following code for your testing, enabling you to use our method on your own dataset.

The whole framework is:

    
