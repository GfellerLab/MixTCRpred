import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import torchvision.transforms as transforms
import os
from src.utils import valid_aa, d_aa_int, q_aa, check_only_standard_aa, clean_aaseq, d_trav_cdr1_homo_sapiens, d_trav_cdr2_homo_sapiens, d_trbv_cdr1_homo_sapiens, d_trbv_cdr2_homo_sapiens, d_trav_cdr1_mus_musculus, d_trav_cdr2_mus_musculus, d_trbv_cdr1_mus_musculus, d_trbv_cdr2_mus_musculus
import itertools

class db_transformer(Dataset):
    def __init__(self, data, data_col = ['epitope', 'cdr3_TRB'], padding =[ 10, 23], epitope = None, istrain= True, host = "HomoSapiens"):
        self.data = data
        self.data_col = ['epitope', 'cdr3_TRA', 'cdr3_TRB']
        if 'epitope' not in self.data.columns:
            self.data['epitope'] = 'XXX'
        len0 = len(self.data)
        #for training/test data: remove nan, duplicate and seq longer than pad
        for idx, col in enumerate(self.data_col):
            if col not in self.data.columns:
                continue
            self.data = self.data.dropna(subset = col, how = 'any')
        #if istrain: #only for train -> rm duplicated seq
        #   self.data = self.data.drop_duplicates(subset = self.data_col)
        for idx, col in enumerate(self.data_col):
            if col not in self.data.columns:
                continue
            self.data = self.data[self.data[col].map(len) <= padding[idx]]
        #check only standard aa
        for idx, col in enumerate(self.data_col):
            if col not in self.data.columns:
                continue
            self.data = clean_aaseq(self.data, col)
            #check_only_standard_aa(self.data, col)
        #epitope specific
        if epitope != None:
            epi_col = self.data_col[0]
            self.data = self.data[self.data[epi_col] == epitope]
        self.seq = self.data[self.data_col].values
        self.len = len(self.data)
        if istrain:
            self.tp = self.data['tp'].values
        if istrain == False:
        #for test set: check if 'tp' in columns (e.g. cross validation),otherwise just set tp=0
            if 'tp' in self.data.columns:
                self.tp = self.data['tp'].values
            else:
                self.tp = [0]*self.len
        self.padding = padding

        ### add vj info
        self.TRAV = self.data['TRAV'].values
        self.TRAJ = self.data['TRAJ'].values
        self.TRBV = self.data['TRBV'].values
        self.TRBJ = self.data['TRBJ'].values

        #mapping V,J gene to cdr12
        self.host = host
        if host == 'HomoSapiens':
            self.map_va_cdr1 = d_trav_cdr1_homo_sapiens
            self.map_va_cdr2 = d_trav_cdr2_homo_sapiens
            self.map_vb_cdr1 = d_trbv_cdr1_homo_sapiens
            self.map_vb_cdr2 = d_trbv_cdr2_homo_sapiens
        if host == 'MusMusculus':
            self.map_va_cdr1 = d_trav_cdr1_mus_musculus
            self.map_va_cdr2 = d_trav_cdr2_mus_musculus
            self.map_vb_cdr1 = d_trbv_cdr1_mus_musculus
            self.map_vb_cdr2 = d_trbv_cdr2_mus_musculus
    def __getitem__(self, index):
        seq_epi_tcr = ""
        seq_epi_tcr_tens = []
        labels = self.tp[index]
        for idx_col, col in enumerate(self.data_col):
            seq = self.seq[index][idx_col]
            len_original = len(seq)
            seq = seq + 'X'*(self.padding[idx_col]-len_original)
            seq_epi_tcr += seq
            #convert to tensor
            seq_tens = torch.tensor(np.array([d_aa_int.get(aa,0) for aa in seq]))
            seq_epi_tcr_tens.append(seq_tens)
        #set V,J genes
        trav = self.TRAV[index]
        traj = self.TRAJ[index]
        trbv = self.TRBV[index]
        trbj = self.TRBJ[index]
        padding_cdr12 = self.padding[-1]
        null_cdr = ''.join(["X"] * padding_cdr12)
        #if nan
        if trav != trav:
            trav = null_cdr
        if trbv != trbv:
            trbv = null_cdr
        #if allele not specified -> consider allele *01
        if "*" not in trav:
            trav = trav+"*01"
        if "*" not in trbv:
            trbv = trbv+"*01"
        #TRAV - CDR1/2
        #TRAV-CDR1
        trav_cdr1 = self.map_va_cdr1.get(trav,null_cdr)
        len_trav_cdr1 = len(trav_cdr1)
        trav_cdr1 = trav_cdr1 + 'X'*(padding_cdr12-len_trav_cdr1)
        trav_cdr1_int = torch.tensor(np.array([d_aa_int.get(aa,0) for aa in trav_cdr1]))
        seq_epi_tcr_tens.append(trav_cdr1_int)
        #TRAV-CDR2
        trav_cdr2 = self.map_va_cdr2.get(trav,null_cdr)
        len_trav_cdr2 = len(trav_cdr2)
        trav_cdr2 = trav_cdr2 + 'X'*(padding_cdr12-len_trav_cdr2)
        trav_cdr2_int = torch.tensor(np.array([d_aa_int.get(aa,0) for aa in trav_cdr2]))
        seq_epi_tcr_tens.append(trav_cdr2_int)
        #TRBV-CDR1
        trbv_cdr1 = self.map_vb_cdr1.get(trbv,null_cdr)
        len_trbv_cdr1 = len(trbv_cdr1)
        trbv_cdr1 = trbv_cdr1 + 'X'*(padding_cdr12-len_trbv_cdr1)
        trbv_cdr1_int = torch.tensor(np.array([d_aa_int.get(aa,0) for aa in trbv_cdr1]))
        seq_epi_tcr_tens.append(trbv_cdr1_int)
        #TRBV-CDR2
        trbv_cdr2 = self.map_vb_cdr2.get(trbv,null_cdr)
        len_trbv_cdr2 = len(trbv_cdr2)
        trbv_cdr2 = trbv_cdr2 + 'X'*(padding_cdr12-len_trbv_cdr2)
        trbv_cdr2_int = torch.tensor(np.array([d_aa_int.get(aa,0) for aa in trbv_cdr2]))
        seq_epi_tcr_tens.append(trbv_cdr2_int)
        seq_epi_tcr = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(seq_epi_tcr, trav, trav_cdr1, trav_cdr2, trbv, trbv_cdr1, trbv_cdr2)
        #print(trav, traj, trbv, trbj)
        #print(trav_cdr1, trav_cdr2, trbv_cdr1, trbv_cdr2)
        return seq_epi_tcr, seq_epi_tcr_tens, labels
    def __len__(self):
        return self.len

class transformer_DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.num_workers= hparams.num_workers
        self.train_path = hparams.train
        self.test_path = hparams.test
        self.batch_size = hparams.batch_size
        self.padding = hparams.padding
        self.epitope = hparams.epitope
        self.host = hparams.host
        if self.train_path != None:
            df = pd.read_csv(self.train_path)
            database = db_transformer(df, padding = self.padding, epitope = self.epitope, host = self.host)
            train_len = int(0.95*database.len) #tried 0.9, 0.99
            valid_len = database.len - train_len
            #print("****", train_len, valid_len)
            self.train, self.valid = random_split(database, lengths=[train_len, valid_len ])
    def train_dataloader(self):
        if self.train_path != None:
            #train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)
            ##add drop_last to assure batch normalization does not give problems: https://stackoverflow.com/questions/65882526/expected-more-than-1-value-per-channel-when-training-got-input-size-torch-size
            train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, drop_last=True)
            return train_loader
    def val_dataloader(self):
        if self.train_path != None:
            valid_loader = DataLoader(self.valid, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)
            return valid_loader
    def test_dataloader(self):
        if self.test_path != None:
            df = pd.read_csv(self.test_path)
            #print(df)
            #self.test = db_transformer(df, padding = self.padding, epitope = self.epitope, istrain = False, host = self.host)
            self.test = db_transformer(df, padding = self.padding, istrain = False, host = self.host)
            test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
            return test_loader




#
#        if self.train_path != None:
#            valid_loader = DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
#            return valid_loader
#    def test_dataloader(self):
#        if self.test_path != None:
#            df = pd.read_csv(self.test_path)
#            #print(df)
#            self.test = db_transformer_MLM(df, self.data_col, padding = self.padding, epitope = self.epitope, istrain = False)
#            test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
#            return test_loader
