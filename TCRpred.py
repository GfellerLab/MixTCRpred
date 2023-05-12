import pandas as pd
import numpy as np
import os
import scipy
from sklearn import metrics
from tqdm import tqdm
from argparse import ArgumentParser
from tabulate import tabulate
import sys
import pathlib
import configparser
import torch
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import src.utils
import src.models
import src.dataloaders


#to supprime all warning message and pytorchlighting info
import warnings
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')


path_pretrained_models = './pretrained_models'


if __name__ == '__main__':


    parser = ArgumentParser(add_help = False)
    ###
    parser.add_argument('--help', action='store_true') #print help
    #model params
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--model', default= None)
    args = parser.parse_args()

    if args.help:
        print("####################################################################################################")
        print("# TCRpred: a sequence-based predictor of TCR-pMHC interaction")
        print("####################################################################################################")
        print("Usage: python TCRpred.py --model [TCRpred_model_name] --input [input_file_of_TCRs] --output [output_file]")
        print("e.g.: python TCRpred.py --model A0201_ELAGIGILTV --input ./test/test.csv --output ./test/output.csv")
        print("")
        print("146 pre-trained TCRpred models are available. Models with less than 50 training TCRs have low confidence")
        df  = pd.read_csv(os.path.join(path_pretrained_models, 'info_models.csv'))
        print(tabulate(df.drop(columns = 'AUC_5fold'), headers='keys', tablefmt='psql'))
        sys.exit(0)

    #################################################
    ##########################################for a quick test
    #args.model = 'A0201_GILGFVFTL'
    #args.input= './test/test.csv'
    #args.output = './test/out_test.csv'
    #########################################################


    ##to make it compatible with previous format
    args.train = None
    args.test = args.input
    args.out = args.output
    args.epitope = args.model.split("_")[-1]
    args.path_checkpoint = os.path.join(path_pretrained_models, 'model_'+args.model+'.ckpt')
    args.epochs = 10
    args.batch_size = 1
    args.num_workers = 1
    args.gpus = None
    args.chain = 'AB'
    #print(args)

    #### determine info epitope
    df_info  = pd.read_csv(os.path.join(path_pretrained_models, 'info_models.csv'))
    df_epi_info = df_info.loc[df_info['Peptide'] == args.epitope]
    model_name = df_epi_info['TCRpred_model_name'].values[0]
    peptide = df_epi_info['Peptide'].values[0]
    origin = df_epi_info['Origin'].values[0]
    MHC_class = df_epi_info['MHC_class'].values[0]
    MHC = df_epi_info['MHC'].values[0]
    n_seq = df_epi_info['Number_training_abTCR'].values[0]
    auc_internal = df_epi_info['AUC_5fold'].values[0]
    host_species = df_epi_info['Host_species'].values[0]
    args.host = host_species


    #set config
    config = configparser.ConfigParser()
    config['transformer'] = {'padding_epitope':len(args.epitope), 'padding_cdr3_TRA':20, 'padding_cdr3_TRB':20, 'padding_cdr12':10, 'embedding_dim':128, 'hidden_dim':1024, 'num_heads':4, 'num_layers':4, 'dropout':0.2, 'lr':1e-4, 'warmup':1000, 'max_iters': 500, 'num_labels':2}
    args.padding = [config['transformer'].getint('padding_epitope'), config['transformer'].getint('padding_cdr3_TRA'), config['transformer'].getint('padding_cdr3_TRB'), config['transformer'].getint('padding_cdr12')]


    #0.load model
    data = src.dataloaders.transformer_DataModule(hparams = args)

    model = src.models.TransformerPredictor_AB_cdr123(
        vocab_size = len(src.utils.valid_aa),
        embedding_dim = config['transformer'].getint('embedding_dim'),
        hidden_dim = config['transformer'].getint('hidden_dim'),
        num_heads = config['transformer'].getint('num_heads'),
        num_layers = config['transformer'].getint('num_layers'),
        dropout = float(config['transformer']['dropout']),
        lr = float(config['transformer']['lr']),
        num_labels = config['transformer'].getint('num_labels'),
        #padding
        padding = args.padding,
        #for better trainig
        warmup = config['transformer'].getint('warmup'),
        max_iters = config['transformer'].getint('max_iters'),
        )

    #1. Set Trainer
    path_model_folder = args.path_checkpoint
    mc = pl.callbacks.ModelCheckpoint(
            dirpath = path_model_folder,
            #filename="transformer-{epoch:02d}-{val_loss:.2f}",
            filename="model_{0}".format(args.epitope), #transformer-{epoch:02d}-{val_loss:.2f}",
            monitor='val_loss'
            )

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir = path_model_folder, max_epochs = int(args.epochs), callbacks = [mc], logger = False) #use logger false to disable tensorboard

    #2. Predictions
    if args.test != None:
        if os.path.exists(args.test) == False:
            print("*** Error! Test file not found. Is --test {0} a valid test file? ***".format(args.test))
            sys.exit(0)
        if (args.path_checkpoint == None) or (os.path.exists(args.path_checkpoint) == False):
            print("*** Error! Model not loaded. Is --path_checkpoint {0} a valid checkpoint? ***".format(args.path_checkpoint))
            sys.exit(0)
        print("#########################################################################")
        print("###### TCRpred: sequence-based predictor of TCR-pMHC interactions  ######")
        print("#########################################################################")
        print("TCRpred model {0} for {1}, ".format(args.model, args.host))
        print("Computing binding predictions for {0}", format(args.test))

        #reload model from checkpoint
        model = model.load_from_checkpoint(
            checkpoint_path= args.path_checkpoint,
            map_location=None,
            )
        trainer.test(model, datamodule = data)
        test_score = model.prob
        test_seq = model.test_seq
        test_tp= model.test_tp
        #add score model to the test set
        df_res = data.test.data
        df_res['score'] = test_score

        #compute perc_rank for a specific epitope
        l_perc_rank = []
        d_par = src.utils.d_par
        for score in df_res['score'].values:
            pr = src.utils.get_perc_rank(score, args.epitope, d_par)
            l_perc_rank.append(pr)
        df_res['perc_rank'] = l_perc_rank
        df_res = df_res.drop('epitope', axis = 1)

        #sort the results
        df_res = df_res.sort_values(by = 'score', ascending = False)

        good_model = False
        if n_seq > 50:
            good_model = True

        f = open(args.output, 'w')
        f.write("##############################################################################################\n")
        f.write("# Output from TCRpred (v1.0)\n")
        f.write("# Predictions of TCRs binding to {0},{1} - class {2} pMHC  \n".format(MHC, peptide, MHC_class.split("MHC")[-1]))
        f.write("# Origin: {0}\n".format(origin))
        f.write("# Input file: {0}\n".format(args.input))
        f.write("# \n")
        f.write("# TCRpred is freely available for academic users.\n")
        f.write("# Private companies should contact XXX at the Ludwig Institute for Cancer Research Ltd for commercial licenses.\n")
        f.write("#\n")
        f.write("# To cite TCRpred,  please refer to: XXXXX\n")
        f.write("#---------------------------------------------------------------------------\n")
        f.write("# Binding predictions with TCRpred model {0}\n".format(model_name))
        if good_model:
            f.write("# Number of training data {0}. Internal AUC {1:.2f}\n".format(n_seq, auc_internal))
        else:
            f.write("# Number of training data {0} (<50!). Low-confidence TCRpred model!! \n".format(n_seq))
        f.write("#---------------------------------------------------------------------------\n")
        f.write("##############################################################################################\n")
        f.close()


        print("The results are stored in {0}.".format(args.out))
        print("If TCRpred is useful for your research, please cite our paper: (XXXX). Thanks!")

        df_res.to_csv(args.out, index = False, mode = 'a')
