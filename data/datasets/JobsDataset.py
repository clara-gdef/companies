import os
import pickle as pkl
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDataset(Dataset):
    def __init__(self, data_dir, rep_type, bag_type):
        print("Loading previously saved dataset...")
        file_name = "jobs_and_emb_wc" + bag_type + "_" + rep_type +"_TEST.pkl"
        with open(os.path.join(data_dir, file_name), 'rb') as f_name:
            dic = torch.load(f_name)
        self.jobs = dic["jobs"]
        self.jobs_emb = dic["job_emb"]
        self.indices = dic["indices"]
        self.preds = dic["pred"]
        self.labels = dic["label"]

        # self.tuples = tmp

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.jobs[idx], self.jobs_emb[idx], self.preds[idx], self.labels[idx]
