import pytorch_lightning as pl
import os
from torch import nn
from torch import autograd
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
from src.utils import valid_aa, d_aa_int, q_aa
import torchvision
import pickle
from sklearn import metrics

class TransformerPredictor_AB_cdr123(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim,  hidden_dim, num_heads, num_layers, dropout, lr, warmup, max_iters, num_labels, padding):
        super(TransformerPredictor_AB_cdr123, self).__init__()
        self.vocab_size =  vocab_size #q_aa
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.max_len = np.sum(padding)
        self.padding_idx = d_aa_int["X"]
        self.warmup = warmup
        self.max_iters= max_iters
        self.lr = lr
        self.prob = []
        self.test_tp = []
        self.test_seq = []
        self.save_hyperparameters()
        #position wise embedding
        self.embedding_tok = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.scale = torch.sqrt(torch.FloatTensor([self.embedding_dim]))
        self.embedding_pos_epi = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[0])
        self.embedding_pos_TRA = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1] + 2*self.padding[-1])
        self.embedding_pos_TRB = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1] + 2*self.padding[-1])
        # Transformer - Encoder
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, input_dim=embedding_dim, dim_feedforward=hidden_dim, num_heads=num_heads, dropout=dropout)
        ### Output classifier
        #self.out_dim = np.sum(self.padding[0:3])*embedding_dim + 4*self.padding[-1]*embedding_dim
        #without epitope
        self.out_dim = np.sum(self.padding[1:3])*embedding_dim + 4*self.padding[-1]*embedding_dim
        self.output_net= nn.Sequential( nn.Linear(self.out_dim,  embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(inplace=True), nn.Linear(embedding_dim, num_labels))
    def forward(self, inp_data, mask=False):
        if mask:
            mask_epi = (inp_data[0] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRA = (inp_data[1] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRB = (inp_data[2] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr1_TRA = (inp_data[3] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr2_TRA = (inp_data[4] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr1_TRB = (inp_data[5] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr2_TRB = (inp_data[6] != self.padding_idx).unsqueeze(1).unsqueeze(2)
        else:
            mask_epi = None
            mask_cdr3_TRA = None
            mask_cdr3_TRB = None
            mask_cdr1_TRA = None
            mask_cdr2_TRA = None
            mask_cdr1_TRB = None
            mask_cdr2_TRB = None
        ####################################################################################################
        ### TRA
        ####################################################################################################
        ### cdr12_TRA
        x_cdr1_TRA = inp_data[3]
        x_cdr2_TRA = inp_data[4]
        x_cdr3_TRA = inp_data[1]
        x_TRA = torch.concat([x_cdr1_TRA, x_cdr2_TRA, x_cdr3_TRA], dim = 1)
        mask_TRA = torch.concat([mask_cdr1_TRA, mask_cdr2_TRA, mask_cdr3_TRA], dim = 3)
        x_emb_TRA = self.embedding_tok(x_TRA)
        x_pos_TRA = self.embedding_pos_TRA(x_TRA)
        scale_TRA = self.scale.to(x_emb_TRA.device)
        x_TRA = (x_emb_TRA * scale_TRA) + x_pos_TRA
        x_TRA_out = self.transformer_encoder(x_TRA, mask=mask_TRA)
        ####################################################################################################
        ### TRB
        ####################################################################################################
        ### cdr12_TRB
        x_cdr1_TRB = inp_data[5]
        x_cdr2_TRB = inp_data[6]
        x_cdr3_TRB = inp_data[2]
        x_TRB = torch.concat([x_cdr1_TRB, x_cdr2_TRB, x_cdr3_TRB], dim = 1)
        mask_TRB = torch.concat([mask_cdr1_TRB, mask_cdr2_TRB, mask_cdr3_TRB], dim = 3)
        x_emb_TRB = self.embedding_tok(x_TRB)
        x_pos_TRB = self.embedding_pos_TRB(x_TRB)
        scale_TRB = self.scale.to(x_emb_TRB.device)
        x_TRB = (x_emb_TRB * scale_TRB) + x_pos_TRB
        x_TRB_out = self.transformer_encoder(x_TRB, mask=mask_TRB)
        ####################################################################################################
        #EPITOPE
        ####################################################################################################
        x_epi = inp_data[0]
        x_emb_epi = self.embedding_tok(x_epi)
        #add positional embedding
        x_pos_epi = self.embedding_pos_epi(x_epi)
        scale_epi = self.scale.to(x_emb_epi.device)
        x_epi = (x_emb_epi * scale_epi) + x_pos_epi
        x_epi_out = self.transformer_encoder(x_epi, mask=mask_epi)
        ##x_epi = x_epi.flatten(start_dim = 1)
        ####################################################################################################
        #concat
        ####################################################################################################
        x = torch.concat([x_TRA_out, x_TRB_out], dim = 1)
        #flatten and classify
        x = x.flatten(start_dim = 1)
        x = self.output_net(x)
        return x
    def configure_optimizers(self):
        #I tried multiple optimizers (Adam, RAdam etc.)
        #best results: Adam + cosine warmup
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if int(self.warmup) > 0:
            self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.warmup, max_iters=self.max_iters)
        return optimizer
    def loss_function(self, out, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, target)
        return loss
    def optimizer_step(self, *args, **kwargs):
        if int(self.warmup) > 0:
            super().optimizer_step(*args, **kwargs)
            self.lr_scheduler.step() # Step per iteration
    def training_step(self, batch, batch_idx):
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.log('train_loss', loss)
        #compute auc
        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
        AUC = metrics.auc(fpr, tpr)
        #print("AUC_train:{0}".format(AUC))
        self.log('train_auc', AUC)
        return loss
    def validation_step(self, batch, batch_idx):
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.log('val_loss', loss)
        #compute auc
        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
        AUC = metrics.auc(fpr, tpr)
        #print("AUC_train:{0}".format(AUC))
        self.log('val_auc', AUC)
        return loss
    def test_step(self, batch, batch_idx):
        test_seq = batch[0]
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.prob.extend(preds.data[:,1].cpu().numpy())
        self.test_tp.extend(labels.cpu().numpy())
        self.test_seq.extend(test_seq)





class TransformerPredictor_AB_cdr123_with_epi(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim,  hidden_dim, num_heads, num_layers, dropout, lr, warmup, max_iters, num_labels, padding):
        super(TransformerPredictor_AB_cdr123, self).__init__()
        self.vocab_size =  vocab_size #q_aa
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.max_len = np.sum(padding)
        self.padding_idx = d_aa_int["X"]
        self.warmup = warmup
        self.max_iters= max_iters
        self.lr = lr
        self.prob = []
        self.test_tp = []
        self.test_seq = []
        self.save_hyperparameters()
        #position wise embedding
        self.embedding_tok = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.scale = torch.sqrt(torch.FloatTensor([self.embedding_dim]))
        self.embedding_pos_epi = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[0])
        self.embedding_pos_cdr3_TRA = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1])
        self.embedding_pos_cdr3_TRB = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[2])
        self.embedding_pos = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, np.sum(self.padding[0:3]))

        self.embedding_pos_cdr12= PositionWiseEmbedding(self.vocab_size, self.embedding_dim, 4*self.padding[-1])



        ########## TEST ############3
        self.embedding_pos_TRA = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1] + 2*self.padding[-1])
        self.embedding_pos_TRB = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1] + 2*self.padding[-1])

        # Transformer - Encoder
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers,
                                              input_dim=embedding_dim,
                                              dim_feedforward=hidden_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        ### Output classifier
        self.out_dim = np.sum(self.padding[0:3])*embedding_dim + 4*self.padding[-1]*embedding_dim

        self.output_net= nn.Sequential(
            nn.Linear(self.out_dim,  embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, num_labels)
            )

    def forward(self, inp_data, mask=False):

        if mask:
            mask_epi = (inp_data[0] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRA = (inp_data[1] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRB = (inp_data[2] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr1_TRA = (inp_data[3] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr2_TRA = (inp_data[4] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr1_TRB = (inp_data[5] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr2_TRB = (inp_data[6] != self.padding_idx).unsqueeze(1).unsqueeze(2)
        else:
            mask_epi = None
            mask_cdr3_TRA = None
            mask_cdr3_TRB = None
            mask_cdr1_TRA = None
            mask_cdr2_TRA = None
            mask_cdr1_TRB = None
            mask_cdr2_TRB = None

        ##### Sequences

        ##EPITOPE
        #x_epi = inp_data[0]
        #x_emb_epi = self.embedding_tok(x_epi)
        ##add positional embedding
        #x_pos_epi = self.embedding_pos_epi(x_epi)
        #scale_epi = self.scale.to(x_emb_epi.device)
        #x_epi = (x_emb_epi * scale_epi) + x_pos_epi
        #x_epi_out = self.transformer_encoder(x_epi, mask=mask_epi)
        ##x_epi = x_epi.flatten(start_dim = 1)
        ##cdr3_TRA
        #x_cdr3_TRA = inp_data[1]
        #x_emb_cdr3_TRA = self.embedding_tok(x_cdr3_TRA)
        #x_pos_cdr3_TRA = self.embedding_pos_cdr3_TRA(x_cdr3_TRA)
        #scale_cdr3_TRA = self.scale.to(x_emb_cdr3_TRA.device)
        #x_cdr3_TRA = (x_emb_cdr3_TRA * scale_cdr3_TRA) + x_pos_cdr3_TRA
        #x_cdr3_TRA_out = self.transformer_encoder(x_cdr3_TRA, mask=mask_cdr3_TRA)
        ##x_cdr3_TRA = x_cdr3_TRA.flatten(start_dim = 1)
        ##cdr3_TRB
        #x_cdr3_TRB = inp_data[2]
        #x_emb_cdr3_TRB = self.embedding_tok(x_cdr3_TRB)
        ##add positional embedding
        #x_pos_cdr3_TRB = self.embedding_pos_cdr3_TRB(x_cdr3_TRB)
        #scale_cdr3_TRB = self.scale.to(x_emb_cdr3_TRB.device)
        #x_cdr3_TRB = (x_emb_cdr3_TRB * scale_cdr3_TRB) + x_pos_cdr3_TRB
        #x_cdr3_TRB_out = self.transformer_encoder(x_cdr3_TRB, mask=mask_cdr3_TRB)
        ##concat and flatten
        #x_cdr3 = torch.concat([x_epi_out, x_cdr3_TRA_out, x_cdr3_TRB_out], dim = 1)

        #### cdr12_TRA
        #x_cdr1_TRA = inp_data[3]
        #x_cdr2_TRA = inp_data[4]
        #### cdr12_TRB
        #x_cdr1_TRB = inp_data[5]
        #x_cdr2_TRB = inp_data[6]
        ##concat
        #x_cdr12 = torch.concat([x_cdr1_TRA, x_cdr2_TRA, x_cdr1_TRB, x_cdr2_TRB], dim = 1)
        #mask_cdr12 = torch.concat([mask_cdr1_TRA, mask_cdr2_TRA, mask_cdr1_TRB, mask_cdr2_TRB ], dim = 3)
        #x_emb_cdr12 = self.embedding_tok(x_cdr12)
        #x_pos_cdr12 = self.embedding_pos_cdr12(x_cdr12)
        #scale_cdr12 = self.scale.to(x_emb_cdr12.device)
        #x_cdr12 = (x_emb_cdr12* scale_cdr12) + x_pos_cdr12
        #x_cdr12 = self.transformer_encoder(x_cdr12, mask=mask_cdr12)
        #### concat seq and v,j info
        #x = torch.concat([x_cdr3, x_cdr12], dim = 1)
        ##add one last transformer encoder
        ##x = self.transformer_encoder(x)#, mask=False)
        ##flatten and classify
        #x = x.flatten(start_dim = 1)
        #x = self.output_net(x)


        ####################################################################################################
        # @!!!!!!! test
        ####################################################################################################


        ####################################################################################################
        ### TRA
        ####################################################################################################
        ### cdr12_TRA
        x_cdr1_TRA = inp_data[3]
        x_cdr2_TRA = inp_data[4]
        x_cdr3_TRA = inp_data[1]
        x_TRA = torch.concat([x_cdr1_TRA, x_cdr2_TRA, x_cdr3_TRA], dim = 1)
        mask_TRA = torch.concat([mask_cdr1_TRA, mask_cdr2_TRA, mask_cdr3_TRA], dim = 3)
        x_emb_TRA = self.embedding_tok(x_TRA)
        x_pos_TRA = self.embedding_pos_TRA(x_TRA)
        scale_TRA = self.scale.to(x_emb_TRA.device)
        x_TRA = (x_emb_TRA * scale_TRA) + x_pos_TRA
        x_TRA_out = self.transformer_encoder(x_TRA, mask=mask_TRA)
        ####################################################################################################
        ### TRB
        ####################################################################################################
        ### cdr12_TRA
        x_cdr1_TRB = inp_data[5]
        x_cdr2_TRB = inp_data[6]
        x_cdr3_TRB = inp_data[2]
        x_TRB = torch.concat([x_cdr1_TRB, x_cdr2_TRB, x_cdr3_TRB], dim = 1)
        mask_TRB = torch.concat([mask_cdr1_TRB, mask_cdr2_TRB, mask_cdr3_TRB], dim = 3)
        x_emb_TRB = self.embedding_tok(x_TRB)
        x_pos_TRB = self.embedding_pos_TRB(x_TRB)
        scale_TRB = self.scale.to(x_emb_TRB.device)
        x_TRB = (x_emb_TRB * scale_TRB) + x_pos_TRB
        x_TRB_out = self.transformer_encoder(x_TRB, mask=mask_TRB)


        ####################################################################################################
        #EPITOPE
        ####################################################################################################
        x_epi = inp_data[0]
        x_emb_epi = self.embedding_tok(x_epi)
        #add positional embedding
        x_pos_epi = self.embedding_pos_epi(x_epi)
        scale_epi = self.scale.to(x_emb_epi.device)
        x_epi = (x_emb_epi * scale_epi) + x_pos_epi
        x_epi_out = self.transformer_encoder(x_epi, mask=mask_epi)
        ##x_epi = x_epi.flatten(start_dim = 1)
        #### FIX THE EPITOPE
        #x_epi_out = torch.zeros((x_epi.shape[0], x_epi.shape[1], x_TRA_out.shape[-1]), device = x_TRA.device)


        #concat
        x = torch.concat([x_epi_out, x_TRA_out, x_TRB_out], dim = 1)
        #add one last transformer encoder
        #x = self.transformer_encoder(x)#, mask=False)
        #flatten and classify
        x = x.flatten(start_dim = 1)
        x = self.output_net(x)
        return x
    def configure_optimizers(self):
        #I tried multiple optimizers (Adam, RAdam etc.)
        #best results: Adam + cosine warmup
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if int(self.warmup) > 0:
            self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.warmup, max_iters=self.max_iters)
        return optimizer
    def loss_function(self, out, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, target)
        return loss
    def optimizer_step(self, *args, **kwargs):
        if int(self.warmup) > 0:
            super().optimizer_step(*args, **kwargs)
            self.lr_scheduler.step() # Step per iteration
    def training_step(self, batch, batch_idx):
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.log('train_loss', loss)

        #compute auc
        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
        AUC = metrics.auc(fpr, tpr)
        #print("AUC_train:{0}".format(AUC))
        self.log('train_auc', AUC)
        return loss
    def validation_step(self, batch, batch_idx):
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.log('val_loss', loss)
        #compute auc
        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
        AUC = metrics.auc(fpr, tpr)
        #print("AUC_train:{0}".format(AUC))
        self.log('val_auc', AUC)
        return loss
    def test_step(self, batch, batch_idx):
        test_seq = batch[0]
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.prob.extend(preds.data[:,1].cpu().numpy())
        self.test_tp.extend(labels.cpu().numpy())
        self.test_seq.extend(test_seq)


class TransformerPredictor_AB_cdr3(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim , hidden_dim, num_heads, num_layers, dropout, lr, warmup, max_iters, num_labels, padding):
        super(TransformerPredictor_AB_cdr3, self).__init__()
        self.vocab_size =  vocab_size #q_aa
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.max_len = np.sum(padding)
        self.padding_idx = d_aa_int["X"]
        self.warmup = warmup
        self.max_iters= max_iters
        self.lr = lr
        self.prob = []
        self.test_tp = []
        self.test_seq = []
        self.save_hyperparameters()
        #position wise embedding
        self.embedding_tok = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.scale = torch.sqrt(torch.FloatTensor([self.embedding_dim]))
        self.embedding_pos_epi = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[0])
        self.embedding_pos_cdr3_TRA = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[1])
        self.embedding_pos_cdr3_TRB = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, self.padding[2])
        self.embedding_pos = PositionWiseEmbedding(self.vocab_size, self.embedding_dim, np.sum(self.padding))
        # Transformer - Encoder
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers,
                                              input_dim=embedding_dim,
                                              dim_feedforward=hidden_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        ## Output classifier
        self.output_net = nn.Sequential(
            nn.Linear(self.max_len * embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, num_labels)
            )
    def forward(self, inp_data, mask=False):
        if mask:
            mask_epi = (inp_data[0] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRA = (inp_data[1] != self.padding_idx).unsqueeze(1).unsqueeze(2)
            mask_cdr3_TRB = (inp_data[2] != self.padding_idx).unsqueeze(1).unsqueeze(2)
        else:
            mask_epi = None
            mask_cdr3_TRA = None
            mask_cdr3_TRB = None
        #EPITOPE
        x_epi = inp_data[0]
        x_emb_epi = self.embedding_tok(x_epi)
        #add positional embedding
        x_pos_epi = self.embedding_pos_epi(x_epi)
        scale_epi = self.scale.to(x_emb_epi.device)
        x_epi = (x_emb_epi * scale_epi) + x_pos_epi
        x_epi_out = self.transformer_encoder(x_epi, mask=mask_epi)
        #x_epi = x_epi.flatten(start_dim = 1)
        #cdr3_TRA
        x_cdr3_TRA = inp_data[1]
        x_emb_cdr3_TRA = self.embedding_tok(x_cdr3_TRA)
        x_pos_cdr3_TRA = self.embedding_pos_cdr3_TRA(x_cdr3_TRA)
        scale_cdr3_TRA = self.scale.to(x_emb_cdr3_TRA.device)
        x_cdr3_TRA = (x_emb_cdr3_TRA * scale_cdr3_TRA) + x_pos_cdr3_TRA
        x_cdr3_TRA_out = self.transformer_encoder(x_cdr3_TRA, mask=mask_cdr3_TRA)
        #x_cdr3_TRA = x_cdr3_TRA.flatten(start_dim = 1)
        #cdr3_TRB
        x_cdr3_TRB = inp_data[2]
        x_emb_cdr3_TRB = self.embedding_tok(x_cdr3_TRB)
        #add positional embedding
        x_pos_cdr3_TRB = self.embedding_pos_cdr3_TRB(x_cdr3_TRB)
        scale_cdr3_TRB = self.scale.to(x_emb_cdr3_TRB.device)
        x_cdr3_TRB = (x_emb_cdr3_TRB * scale_cdr3_TRB) + x_pos_cdr3_TRB
        x_cdr3_TRB_out = self.transformer_encoder(x_cdr3_TRB, mask=mask_cdr3_TRB)
        #x_cdr3_TRB = x_cdr3_TRB.flatten(start_dim = 1)
        #####x = x.flatten(start_dim = 1)
        x = torch.concat([x_epi_out, x_cdr3_TRA_out, x_cdr3_TRB_out], dim = 1)
        x = x.flatten(start_dim = 1)
        x = self.output_net(x)
        return x
    def configure_optimizers(self):
        #I tried multiple optimizers (Adam, RAdam etc.)
        #best results: Adam + cosine warmup
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        if int(self.warmup) > 0:
            self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.warmup, max_iters=self.max_iters)
        return optimizer
    def loss_function(self, out, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, target)
        return loss
    def optimizer_step(self, *args, **kwargs):
        if int(self.warmup) > 0:
            super().optimizer_step(*args, **kwargs)
            self.lr_scheduler.step() # Step per iteration
    def training_step(self, batch, batch_idx):
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.log('train_loss', loss)
        #compute auc
        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
        AUC = metrics.auc(fpr, tpr)
        #print("AUC_train:{0}".format(AUC))
        self.log('train_auc', AUC)
        return loss
    def validation_step(self, batch, batch_idx):
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.log('val_loss', loss)
        #compute auc
        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
        AUC = metrics.auc(fpr, tpr)
        #print("AUC_train:{0}".format(AUC))
        self.log('val_auc', AUC)
        return loss
    def test_step(self, batch, batch_idx):
        test_seq = batch[0]
        inp_data = batch[1]
        labels = batch[2]
        preds = self.forward(inp_data, mask = True)
        loss = self.loss_function(preds, labels)
        self.prob.extend(preds.data[:,1].cpu().numpy())
        self.test_tp.extend(labels.cpu().numpy())
        self.test_seq.extend(test_seq)



#class combined_model(pl.LightningModule):
#    def __init__(self, all_trained_model, all_hparams, all_state_dict):
#        super(combined_model, self).__init__()
#        #load weights
#        self.n_model = len(all_trained_model)
#        self.all_models = []
#        for model_trained, model_hparams, model_state_dict in zip(all_trained_model, all_hparams, all_state_dict):
#            #print(model_hparams)
#            model_seq = model_trained#(**model_hparams)
#            model_seq.load_state_dict(model_state_dict)
#            model_seq.freeze()
#            self.all_models.append(model_seq)
#        self.classifier = torch.nn.Linear(self.n_model*2, 2)
#        #self.save_hyperparameters()
#        ## store predictions
#        self.prob = []
#        self.test_tp = []
#        self.test_seq = []
#    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
#        return optimizer
#    def loss_function(self, out, target):
#        loss_fn = nn.CrossEntropyLoss()
#        loss = loss_fn(out, target)
#        return loss
#    def forward(self, inp_data):
#        all_out = []
#        for x_idx, x in enumerate(inp_data):
#            out = self.all_models[x_idx].to(inp_data[0].device)(x.to(inp_data[0].device))
#            all_out.append(out)
#        x = torch.cat(all_out, dim=1)
#        x = self.classifier(x)
#        return x
#    def training_step(self, batch, batch_idx):
#        inp_data = batch[2]
#        labels = batch[3]
#        preds = self.forward(inp_data)
#        loss = self.loss_function(preds, labels)
#        return loss
#    def validation_step(self, batch, batch_idx):
#        inp_data = batch[2]
#        labels = batch[3]
#        preds = self.forward(inp_data)
#        loss = self.loss_function(preds, labels)
#        self.log('val_loss', loss)
#        return loss
#    def test_step(self, batch, batch_idx):
#        test_seq = batch[0]
#        inp_data = batch[2]
#        labels = batch[3]
#        preds = self.forward(inp_data)
#        self.prob.extend(preds.data[:,1].cpu().numpy())
#        self.test_tp.extend(labels.cpu().numpy())
#        self.test_seq.extend(test_seq)



#####################################################################################################
## TRANSFORMERS
##https://pytorchlightning.github.io/lightning-tutorials/notebooks/course_UvA-DL/transformers-and-MH-attention.html
##also here (easy implementation of attention)
##https://github.com/idiap/fast-transformers
##https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
#

#class TransformerPredictor(pl.LightningModule):
#    def __init__(self, vocab_size, embedding_dim , hidden_dim, num_heads, num_layers, dropout, lr, warmup, max_iters, num_labels, padding):
#        super(TransformerPredictor, self).__init__()
#        self.vocab_size =  vocab_size #q_aa
#        self.embedding_dim = embedding_dim
#        self.padding = padding
#        self.padding_idx = d_aa_int["X"]
#        self.warmup = warmup
#        self.max_iters= max_iters
#        self.lr = lr
#        self.prob = []
#        self.test_tp = []
#        self.test_seq = []
#        self.save_hyperparameters()
#        #position wise embedding
#        self.embedding_tok = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.padding_idx)
#        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))
#        self.embedding_pos = PositionWiseEmbedding(self.vocab_size, embedding_dim, self.max_len)
#        # Transformer
#        self.transformer = TransformerEncoder(num_layers=num_layers,
#                                              input_dim=embedding_dim,
#                                              dim_feedforward=hidden_dim,
#                                              num_heads=num_heads,
#                                              dropout=dropout)
#        ## Output classifier
#        self.output_net = nn.Sequential(
#            #first NN
#            nn.Linear(self.max_len * embedding_dim, embedding_dim),
#            nn.LayerNorm( embedding_dim),
#            nn.ReLU(inplace=True),
#            nn.Dropout(dropout),
#            nn.Linear(embedding_dim, num_labels)
#            )
#    def forward(self, x, mask=None):
#        # for each seq -> compute self attention
#        for x in  batch[1]:
#            x_emb = self.embedding_tok(x)
#            #add positional embedding
#            x_pos = self.embedding_pos(x)
#            scale = self.scale.to(x_emb.device)
#            x = (x_emb * scale) + x_pos
#            x = self.transformer(x, mask=mask)
#            break
#        #print(x.shape)
#        x = x.flatten(start_dim = 1)
#        #print(x.shape)
#        x = self.output_net(x)
#        return x
#    def configure_optimizers(self):
#        #I tried multiple optimizers (Adam, RAdam etc.)
#        #best results: Adam + cosine warmup
#        optimizer = optim.Adam(self.parameters(), lr=self.lr)
#        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
#        if int(self.warmup) > 0:
#            self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.warmup, max_iters=self.max_iters)
#        return optimizer
#    def loss_function(self, out, target):
#        loss_fn = nn.CrossEntropyLoss()
#        loss = loss_fn(out, target)
#        return loss
#    def optimizer_step(self, *args, **kwargs):
#        if int(self.warmup) > 0:
#            super().optimizer_step(*args, **kwargs)
#            self.lr_scheduler.step() # Step per iteration
#    def training_step(self, batch, batch_idx):
#        labels = batch[3]
#        preds = self.forward(inp_data, mask = mask_padding)
#        loss = self.loss_function(preds, labels)
#        self.log('train_loss', loss)
#        #compute auc
#        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
#        AUC = metrics.auc(fpr, tpr)
#        #print("AUC_train:{0}".format(AUC))
#        self.log('train_auc', AUC)
#        return loss
#    def validation_step(self, batch, batch_idx):
#        inp_data = batch[2][self.n_ngrams-1]
#        labels = batch[3]
#        mask_padding = None
#        if self.mask_pad:
#            tmp_matrix = np.zeros(inp_data.shape)
#            for idx_mask in self.idx_to_mask:
#                tmp_matrix += ((inp_data == idx_mask) + 0).cpu().numpy()
#            mask_padding = torch.tensor(tmp_matrix).unsqueeze(1).unsqueeze(2).to(inp_data.device)
#        preds = self.forward(inp_data, mask = mask_padding)
#        loss = self.loss_function(preds, labels)
#        self.log('val_loss', loss)
#        #compute auc
#        fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), preds.data[:,1].cpu().numpy())
#        AUC = metrics.auc(fpr, tpr)
#        #print("AUC_train:{0}".format(AUC))
#        self.log('val_auc', AUC)
#        return loss
#    def test_step(self, batch, batch_idx):
#        test_seq = batch[0]
#        test_token = batch[1]
#        inp_data = batch[2][self.n_ngrams-1]
#        labels = batch[3]
#        mask_padding = None
#        if self.mask_pad:
#            tmp_matrix = np.zeros(inp_data.shape)
#            for idx_mask in self.idx_to_mask:
#                tmp_matrix += ((inp_data == idx_mask) + 0).cpu().numpy()
#            mask_padding = torch.tensor(tmp_matrix).unsqueeze(1).unsqueeze(2).to(inp_data.device)
#        preds = self.forward(inp_data, mask = mask_padding)
#        self.prob.extend(preds.data[:,1].cpu().numpy())
#        self.test_tp.extend(labels.cpu().numpy())
#        self.test_seq.extend(test_seq)
#        #for attention
#        #attn = self.get_attention_maps(inp_data, mask = mask_padding)
#        #self.test_attention.extend(attn)#.cpu().numpy())








def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]
    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels([d_int_aa[i] for i in input_data.tolist()])
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels([d_int_aa[i] for i in input_data.tolist()])
            ax[row][column].set_title("Layer %i, Head %i" % (row+1, column+1))
    fig.subplots_adjust(hspace=0.5)


#### auxiliary functions for transformers

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

### decoder

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps



class PositionalEncoding(nn.Module):
    def __init__(self, q_aa, d_model, max_len=70):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor




####################################################################################################
# a learned position-wise encoding (used also in BERT, better than Attention-is-all-you-need pos encoding
####################################################################################################
#class PositionWiseEmbedding(nn.Module):
#    def __init__(self, vocab_size, embedding_dim, padding_idx, dropout_p, padding): #max length -> see padding parameters
#        super().__init__()
#        self.embedding_dim = embedding_dim
#        self.max_len = np.sum(padding)
#        self.dropout_p = dropout_p
#        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
#        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
#        self.dropout = nn.Dropout(dropout_p)
#        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))
#    def forward(self, x):
#        # inputs = [batch size, inputs len]
#        batch_size = x.shape[0]
#        inputs_len = x.shape[1]
#        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
#        scale = self.scale.to(x.device)
#        embedded = (self.tok_embedding(x) * scale) + self.pos_embedding(pos)
#        # output = [batch size, inputs len, hid dim]
#        output = self.dropout(embedded)
#        return output

def compute_len_max(padding, n_ngrams):
    L_seq = np.sum(padding)
    print('**************', L_seq, n_ngrams)
    tot_len = L_seq - n_ngrams + 1
    return tot_len

class PositionWiseEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len): #max length -> see padding parameters
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.pos_embedding = nn.Embedding(self.max_len, embedding_dim)
    def forward(self, x):
        # inputs = [batch size, inputs len]
        batch_size = x.shape[0]
        inputs_len = x.shape[1]
        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        pos_embedding = self.pos_embedding(pos)
        return pos_embedding

        #scale = self.scale.to(x.device)
        #self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))
        #embedded = (self.tok_embedding(x) * scale) + self.pos_embedding(pos)


############## ANOTHER MODEL
#check here: https://peterbloem.nl/blog/transformers

#class PositionalEncoding2(nn.Module):
#    def __init__(self, hid_dim, max_len = 65): #max length -> see padding parameters
#        super().__init__()
#        self.hid_dim = hid_dim
#        self.max_len = max_len
#        self.pos_embedding = nn.Embedding(max_len, hid_dim)
#        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
#    def forward(self, x):
#        # inputs = [batch size, inputs len]
#        batch_size = x.shape[0]
#        inputs_len = x.shape[1]
#        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
#        scale = self.scale.to(x.device)
#        embedded = self.pos_embedding(pos)
#        return embedded
#
#class PositionalEncoding2(nn.Module):
#    def __init__(self, vocab_size, embedding_dim, max_len = 65): #max length -> see padding parameters
#        super().__init__()
#        self.embedding_dim = embedding_dim
#        self.max_len = max_len
#        #self.dropout_p = dropout_p
#        #https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
#        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
#        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
#        #self.dropout = nn.Dropout(dropout_p)
#        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))
#    def forward(self, x):
#        # inputs = [batch size, inputs len]
#        batch_size = x.shape[0]
#        inputs_len = x.shape[1]
#        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
#        scale = self.scale.to(x.device)
#        embedded = (self.tok_embedding(x) * scale) + self.pos_embedding(pos)
#        # output = [batch size, inputs len, hid dim]
#        #output = self.dropout(embedded)
#        return embedded
#
#class TransformerNet(pl.LightningModule):
#    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, max_len, num_labels, dropout, lr):
#        super(TransformerNet, self).__init__()
#        self.lr = lr
#        self.prob = []
#        self.test_tp = []
#        self.test_seq = []
#        # embedding layer
#        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
#        # positional encoding layer
#        self.pe = PositionalEncoding2(vocab_size, embedding_dim, max_len)
#        # encoder  layers
#        enc_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout, batch_first = True)
#        self.encoder = nn.TransformerEncoder(enc_layer, num_layers = num_layers)
#        # final dense layer
#        self.dense = nn.Linear(embedding_dim*max_len, num_labels)
#        self.log_softmax = nn.LogSoftmax()
#        # Output classifier
#        #self.output_net = nn.Sequential(
#        #    nn.Linear(max_len * embedding_dim, self.hparams.model_dim),
#        #    nn.LayerNorm(model_dim),
#        #    nn.ReLU(inplace=True),
#        #    nn.Dropout(dropout),
#        #    nn.Linear(model_dim, num_labels)
#        #)
#    def forward(self, x):
#        #x = self.embedding(x)#.permute(1, 0, 2) # -> check this permutation
#        x = self.pe(x)
#        x = self.encoder(x)
#        #x = x.reshape(x.shape[1], -1)
#        x = x.flatten(start_dim = 1)
#        x = self.dense(x)
#        return x
#    def configure_optimizers(self):
#        #I tried multiple optimizers (Adam, RAdam etc.)
#        #best results: Adam
#        optimizer = optim.Adam(self.parameters(), lr=self.lr)
#        return optimizer
#    def loss_function(self, out, target):
#        loss_fn = nn.CrossEntropyLoss()
#        loss = loss_fn(out, target)
#        return loss
#    def training_step(self, batch, batch_idx):
#        inp_data = batch[1]
#        #pad_idx = 0
#        #mask_padding = (inp_data != pad_idx).unsqueeze(1).unsqueeze(2) # masking the padded index doesn't improve the results!
#        mask_padding = None
#        labels = batch[2]
#        preds = self.forward(inp_data)#, mask = mask_padding)
#        loss = self.loss_function(preds, labels)
#        self.log('train_loss', loss)
#        return loss
#    def validation_step(self, batch, batch_idx):
#        inp_data = batch[1]
#        #pad_idx = 0
#        #mask_padding = (inp_data != pad_idx).unsqueeze(1).unsqueeze(2)
#        mask_padding = None
#        labels = batch[2]
#        preds = self.forward(inp_data)#, mask = mask_padding)
#        loss = self.loss_function(preds, labels)
#        self.log('val_loss', loss)
#        return loss
#    def test_step(self, batch, batch_idx):
#        test_seq = batch[0]
#        inp_data = batch[1]
#        labels = batch[2]
#        #pad_idx = 0
#        #mask_padding = (inp_data != pad_idx).unsqueeze(1).unsqueeze(2)
#        mask_padding = None
#        preds = self.forward(inp_data)#, mask = mask_padding)
#        self.prob.extend(preds.data[:,1].cpu().numpy())
#        self.test_tp.extend(labels.cpu().numpy())
#        self.test_seq.extend(test_seq)
#        #for attention
#        #attn = self.get_attention_maps(inp_data, mask = mask_padding)
#        #self.test_attention.extend(attn)#.cpu().numpy())
#
#
#
#
#
