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
