import sys
from finetune import *
from modelepi import *
from readfile import *
from function import *
from finetunecross import *
from pretrain import *
from motif import *
from text import *

def main():
    if len(sys.argv)<2:
        text_for_print(1)
        return
    else:
        if sys.argv[1]=="pretraining":
            if len(sys.argv)!=8:
                text_for_print(2)
            else:
                print("This is the pretraining part of the software.")
                lf_pre_training(sys.argv[2], sys.argv[3], sys.argv[4], tokenizer_maxlen=int(sys.argv[7]), batchsize=int(sys.argv[5]), num_hidden_layers=int(sys.argv[6]))
        elif sys.argv[1]=="rbpretraining":
            if len(sys.argv)!=8:
                text_for_print(3)
            else:
                print("This is the pretraining part of the software.")
                robert_lf_pre_training(sys.argv[2], sys.argv[3], sys.argv[4], tokenizer_maxlen=int(sys.argv[7]), batchsize=int(sys.argv[5]), num_hidden_layers=int(sys.argv[6]))
        elif sys.argv[1]=="finetuningbertcnntrain":
            if len(sys.argv)!=10:
                text_for_print(4)
            else:
                print("This is the fine tuning part of the software.")
                lf_fine_tuning_bertcnn_train(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], train_batchsize=int(sys.argv[8]), learning_rate=2e-5, epochs=int(sys.argv[9]))
        elif sys.argv[1]=="finetuningbertcnnvalidate":
            if len(sys.argv)!=9:
                text_for_print(5)
            else:
                print("This is the fine tuning part of the software.")
                lf_fine_tuning_bertcnn_validate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], validate_batchsize=int(sys.argv[7]), epochs=int(sys.argv[8]))
        elif sys.argv[1]=="finetuningbertcnntranstrain":
            if len(sys.argv)!=12:
                text_for_print(6)
            else:
                print("This is the fine tuning part of the software.")
                lf_fine_tuning_bertcnn_trans_train(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], train_batchsize=int(sys.argv[8]), validate_batchsize=int(sys.argv[9]), learning_rate=2e-5, kernel_size=int(sys.argv[10]), epochs=int(sys.argv[11]))
        elif sys.argv[1]=="finetuningbertcnntransvalidate":
            if len(sys.argv)!=9:
                text_for_print(7)
            else:
                print("This is the fine tuning part of the software.")
                lf_fine_tuning_bertcnn_trans_validate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], validate_batchsize=int(sys.argv[7]), epochs=int(sys.argv[8]))
        elif sys.argv[1]=="motifdiscovery":
            if len(sys.argv)!=9:
                text_for_print(8)
            else:
                print("To discover motifs based on the attention matrix of Transformer.")
                motif_discovery(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
        elif sys.argv[1]=="interactkmerdiscovery":
            if len(sys.argv)!=9:
                text_for_print(9)
            else:
                print("To discover motifs based on the attention matrix of Transformer.")
                discover_kmer_interaction(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7]), int(sys.argv[8]))
        else:
            print("Error! Please select a correct module")
            return
    
if __name__ == '__main__':
    main()

    