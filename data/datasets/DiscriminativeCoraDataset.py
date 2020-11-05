import os
import pickle as pkl
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativeCoraDataset(Dataset):
    def __init__(self, datadir, paper_file, track_file, split, subsample, load):
        self.datadir = datadir
        if load:
            self.load_dataset()
        else:
            self.split = split

            self.build_tuples(paper_file)
            self.build_bag_reps(track_file)
            self.save_dataset()


    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

    def save_dataset(self):
        pass

    def load_dataset(self):
        pass

    def build_tuples(self, paper_file, split):
        pass

    def build_tuples(self, track_file):
        pass