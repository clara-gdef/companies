import os
import pickle as pkl
import ipdb
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativeCoraDataset(Dataset):
    def __init__(self, datadir, paper_file, bag_file, split, subsample, ft_type, high_level, load):
        self.datadir = datadir
        self.ft_type = ft_type
        self.split = split
        self.high_level = high_level
        if load:
            self.load_dataset()
        else:
            self.tuples = []
            self.ft_type = ft_type
            self.split = split
            self.build_tuples(os.path.join(datadir, paper_file + "_" + ft_type + "_" + split + ".pkl"))
            self.build_bag_reps(os.path.join(datadir, bag_file + "_" + ft_type + ".pkl"))
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
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        if self.high_level:
            tgt_file = os.path.join(self.datadir, "cora_hl_dataset_" + self.ft_type + "_" + self.split + '.pkl')
        else:
            tgt_file = os.path.join(self.datadir, "cora_dataset_" + self.ft_type + "_" + self.split + '.pkl')
        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)
        print("Dataset saved : " + tgt_file)

    def load_dataset(self):
        if self.high_level:
            tgt_file = os.path.join(self.datadir, "cora_hl_dataset_" + self.ft_type + "_" + self.split + '.pkl')
        else:
            tgt_file = os.path.join(self.datadir, "cora_dataset_" + self.ft_type + "_" + self.split + '.pkl')
        with open(tgt_file, 'rb') as f:
            dico = pkl.load(f)
        for key in tqdm(dico, desc="Loading attributes from save..."):
            vars(self)[key] = dico[key]
        print("Dataset load from : " + tgt_file)

    def build_tuples(self, paper_file):
        if self.high_level:
            mapper_dict = pkl.load(open(os.path.join(self.datadir, "cora_track_to_hl_classes_map.pkl"), 'rb'))
        # length that keeps 90% of the dataset untrimmed
        max_abstract_len = 12
        with open(paper_file, 'rb') as f:
            data = pkl.load(f)
        for tup in tqdm(data, desc="Building tuples for split " + self.split + '...'):
            new_tup = {}
            for k in ['id', 'avg_profile']:
                new_tup[k] = tup[k]
            if self.high_level:
                new_tup["class"] = mapper_dict[tup["class"]]
            else:
                new_tup["class"] = tup["class"]
            sent_emb = np.zeros((max_abstract_len, 300))
            for num, sent in enumerate(tup["sentences_emb"]):
                if num < max_abstract_len:
                    sent_emb[num, :] = sent
            new_tup["sentences_emb"] = sent_emb
            self.tuples.append(new_tup)

    def build_bag_reps(self, track_file):
        with open(track_file, 'rb') as f:
            tracks = pkl.load(f)
        self.track_rep = torch.from_numpy(np.stack(tracks))
