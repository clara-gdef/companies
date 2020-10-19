import os
import pickle as pkl
import itertools

import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDatasetPoly(Dataset):
    def __init__(self, data_dir, bag_type, load, bag_file, subsample, split):
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(data_dir, split)
        else:
            with open(os.path.join(data_dir, bag_file), "rb") as f_name:
                bag_reps = pkl.load(f_name)
            tgt_file = os.path.join(data_dir, "JobsDatasetPoly_" + split + ".pkl")
            with open(tgt_file, 'rb') as f:
                ds_dict = pkl.load(f)

            self.rep_dim = 300
            self.num_bags = len(bag_reps)
            self.bag_reps = bag_reps
            self.tuples = []
            self.select_relevant_tuples(ds_dict, bag_type, subsample)
            self.save_dataset(data_dir, split)

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"], self.tuples[idx]["rep"], self.tuples[idx]["cie"], self.tuples[idx]["clus"], self.tuples[idx]["dpt"], self.bag_rep


    def save_dataset(self, datadir, split):
        ds_dict = {"rep_dim": self.rep_dim,
                   "num_bags": self.num_bags,
                   "bag_rep": self.bag_rep,
                   "tuples": self.tuples}
        tgt_file = os.path.join(datadir, "JobsDatasetSpe_" + split + ".pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(ds_dict, f)

    def load_dataset(self, datadir, split):
        tgt_file = os.path.join(datadir, "JobsDatasetSpe_" + split + ".pkl")
        with open(tgt_file, 'rb') as f:
            ds_dict = pkl.load(f)
        self.rep_dim = ds_dict["rep_dim"]
        self.num_bag = ds_dict["num_bags"]
        self.bag_rep = ds_dict["bag_rep"]
        self.tuples = ds_dict["tuples"]

    def select_relevant_tuples(self, all_tuples, bag_type, subsample):
        tmp = []
        for person in all_tuples:
            tmp.append({"id": person["id"],
                        "rep": person["rep"],
                        "jobs_len": person["jobs_len"],
                        "bag_rep": self.bag_rep,
                        "label": person[bag_type]
                        })
            if subsample > 0:
                np.random.shuffle(self.tuples)
                self.tuples = tmp[:subsample]
            else:
                self.tuples = tmp

