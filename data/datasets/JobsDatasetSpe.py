import os
import pickle as pkl
import itertools

import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDatasetSpe(Dataset):
    def __init__(self, data_dir, bag_type, load, bag_file, subsample, split, standardized):
        if load == "True":
            print("Loading previously saved dataset...")
            self.load_dataset(data_dir, split, standardized)
        else:

            if standardized:
                tgt_file = os.path.join(data_dir, "JobsDatasetPoly_" + split + "_standardized.pkl")
                with open(tgt_file, 'rb') as f:
                    ds_dict = pkl.load(f)
                with open(os.path.join(data_dir, bag_file + "_standardized.pkl"), "rb") as f_name:
                    bag_reps = pkl.load(f_name)
            else:
                tgt_file = os.path.join(data_dir, "JobsDatasetPoly_" + split + ".pkl")
                with open(tgt_file, 'rb') as f:
                    ds_dict = pkl.load(f)
                with open(os.path.join(data_dir, bag_file + ".pkl"), "rb") as f_name:
                    bag_reps = pkl.load(f_name)

            self.rep_dim = 300
            self.num_bags = len(bag_reps)
            self.bag_reps = self.build_bag_tensor(bag_reps)
            self.tuples = []
            self.select_relevant_tuples(ds_dict["tuples"], bag_type, subsample)
            self.save_dataset(data_dir, split, standardized)
        if subsample > 0:
            np.random.shuffle(self.tuples)
            tmp = self.tuples[:subsample]
            self.tuples = tmp
        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"], self.tuples[idx]["rep"], self.tuples[idx]["jobs_len"], self.tuples[idx]["label"], self.bag_reps

    def build_bag_tensor(self, bag_dict):
        bags = torch.zeros(1, 300)
        for k in bag_dict.keys():
            bags = torch.cat((bags, bag_dict[k]["ft"]), dim=0)
        return bags[1:]

    def save_dataset(self, datadir, split, standardized):
        ds_dict = {"rep_dim": self.rep_dim,
                   "num_bags": self.num_bags,
                   "bag_reps": self.bag_reps,
                   "tuples": self.tuples}
        if standardized:
            tgt_file = os.path.join(datadir, "JobsDatasetSpe_" + split + "_standardized.pkl")
            with open(tgt_file, 'wb') as f:
                pkl.dump(ds_dict, f)
        else:
            tgt_file = os.path.join(datadir, "JobsDatasetSpe_" + split + ".pkl")
            with open(tgt_file, 'wb') as f:
                pkl.dump(ds_dict, f)

    def load_dataset(self, datadir, split, standardized):
        if standardized:
            tgt_file = os.path.join(datadir, "JobsDatasetSpe_" + split + "_standardized.pkl")
            with open(tgt_file, 'rb') as f:
                ds_dict = pkl.load(f)
        else:
            tgt_file = os.path.join(datadir, "JobsDatasetSpe_" + split + ".pkl")
            with open(tgt_file, 'rb') as f:
                ds_dict = pkl.load(f)
        self.rep_dim = ds_dict["rep_dim"]
        self.num_bags = ds_dict["num_bags"]
        self.bag_reps = ds_dict["bag_reps"]
        self.tuples = ds_dict["tuples"]

    def select_relevant_tuples(self, all_tuples, bag_type, subsample):
        tmp = []
        for person in tqdm(all_tuples, desc="Selecting relevant tuples..."):
            tmp.append({"id": person["id"],
                        "rep": person["rep"],
                        "jobs_len": person["jobs_len"],
                        "bag_rep": self.bag_reps,
                        "label": person[bag_type]
                        })
            if subsample > 0:
                np.random.shuffle(self.tuples)
                self.tuples = tmp[:subsample]
            else:
                self.tuples = tmp

