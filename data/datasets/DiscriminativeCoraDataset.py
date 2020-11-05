import os
import pickle as pkl
import ipdb
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativeCoraDataset(Dataset):
    def __init__(self, datadir, paper_file, track_file, split, subsample, ft_type, load):
        self.datadir = datadir
        self.tuples = []
        if load:
            self.load_dataset()
        else:
            self.ft_type = ft_type
            self.split = split
            self.build_tuples(os.path.join(datadir, paper_file + "_" + ft_type + "_" + split + ".pkl"))
            self.build_bag_reps(os.path.join(datadir, track_file + "_" + ft_type + ".pkl"))
            self.save_dataset()

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"], \
               self.tuples[idx]["avg_profile"], \
               self.tuples[idx]["sentences_emb"],\
               self.tuples[idx]["class"], \
               self.track_rep

    def save_dataset(self):
        dico = {}
        ipdb.set_trace()
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico = {str(attribute): self.attribute}
        tgt_file = os.path.join(self.datadir, "cora_dataset_" + self.ft_type + "_" + self.split + '.pkl')
        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)

    def load_dataset(self):
        pass

    def build_tuples(self, paper_file):
        with open(paper_file, 'rb') as f:
            data = pkl.load(f)
        ipdb.set_trace()
        for tup in tqdm(data, desc="Building tuples for split " + self.split + '...'):
            self.tuples.append(tup)

    def build_tuples(self, track_file):
        with open(track_file, 'rb') as f:
            tracks = pkl.load(f)
        self.track_rep = torch.from_numpy(tracks)
