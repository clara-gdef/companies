import os
import pickle as pkl
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDataset(Dataset):
    def __init__(self, data_dir, rep_type, bag_type, bag_rep):
        self.jobs = []
        self.jobs_emb = []
        self.indices = []
        self.preds = []
        self.labels = []
        self.bag_rep = bag_rep
        print("Loading previously saved dataset...")
        file_name = "jobs_and_emb_wc_" + bag_type + "_" + rep_type +"_TEST.pkl"
        with open(os.path.join(data_dir, file_name), 'rb') as f_name:
            dic = torch.load(f_name)
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
