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
    random
    matplotlib
    scikit-learn
    transformers 4.29.2
    pathlib
    os
    pickle
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
Here the tokenizer path is the path of tokenizer. You are suggested to use the tokenizer under the folder tokenizer. 
The input dir path is the folder path of DNA sequences.
The model save path is the path you used to save the pretrained model.
The batch size : Default 2
Number hidden layers : Number of transformer encoders of your pretrained model. Default 4
Max tokenizer length : Max tokens of your input sequences. Default: 5100

If you do not want to pretrain a model by yourself, we prepare a pretrained model under the folder pretrained_model.

### Cell type-specific fine-tuning
This module is used to fine-tune a model with your own enhancer promoter pairs. 
