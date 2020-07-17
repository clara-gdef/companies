import os
import pickle as pkl
import torch
from torch.utils.data import Dataset


class DicriminativePolyvalentDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datadir, ppl_file, agg_type, cie_reps_file, clus_reps_file, dpt_reps_file):
        print("Loading data...")
        with open(os.path.join(datadir, ppl_file), 'rb') as f:
            ppl_dict = pkl.load(f)
        with open(os.path.join(datadir, cie_reps_file + "_" + agg_type + ".pkl"), "rb") as f:
            cie_reps = pkl.load(f)
        with open(os.path.join(datadir, clus_reps_file + "_" + agg_type + ".pkl"), "rb") as f:
            clus_reps = pkl.load(f)
        with open(os.path.join(datadir, dpt_reps_file + "_" + agg_type + ".pkl"), "rb") as f:
            dpt_reps = pkl.load(f)
        print("Data Loaded.")


    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]