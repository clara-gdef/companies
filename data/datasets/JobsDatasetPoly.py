import os
import pickle as pkl
import itertools

import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDatasetPoly(Dataset):
    def __init__(self, data_dir, cie_reps_file, clus_reps_file, dpt_reps_file, load, split):
        self.jobs = []
        self.jobs_emb = []
        self.indices = []
        self.preds = []
        self.labels = []
        print("Loading previously saved dataset...")
        file_name = "total_rep_jobs_unflattened_" + split + ".pkl"
        with open(os.path.join(data_dir, file_name), 'rb') as f_name:
            dic = torch.load(f_name)
        ipdb.set_trace()
        for i in dic.keys():
            self.jobs.append(dic[i]["jobs"])
            self.jobs_emb.append(dic[i]["job_emb"])
            self.indices.append(i)
            self.preds.append(dic[i]["pred"])
            self.labels.append(dic[i]["label"])

        # self.tuples = tmp

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.jobs[idx], self.jobs_emb[idx], self.preds[idx], self.labels[idx], self.bag_rep
