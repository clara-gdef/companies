import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import pickle as pkl
from utils.models import labels_to_one_hot
from sklearn.metrics import confusion_matrix, classification_report


class DebugClassifierDisc(torch.nn.Module):

    def __init__(self, in_size, out_size, hparams, dataset, datadir, desc):
        super().__init__()
        self.lin = torch.nn.Linear(in_size, out_size)
        torch.nn.init.eye_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)


        self.training_losses = []
        self.test_outputs = []
        self.test_labels_one_hot = []
        self.test_labels = []
        self.test_ppl_id = []
        self.type = desc.split("_")[1]
        if self.type == 'spe':
            self.bag_type = desc.split("_")[2]

        self.input_type = hparams.input_type
        self.num_cie = dataset.num_cie
        self.num_clus = dataset.num_clus
        self.num_dpt = dataset.num_dpt

        self.hparams = hparams
        self.data_dir = datadir
        self.description = desc

        ### debug
        self.before_training = []

    def forward(self, x):
        return self.lin(x)

