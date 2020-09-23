import os
import pickle as pkl
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativeSpecializedDataset(Dataset):
    def __init__(self, data_dir, rep_type, agg_type, bag_type, split, subsample):
        print("Loading previously saved dataset...")
        file_name = "disc_poly_" + agg_type + "_" + rep_type + "_" + split + ".pkl"
        with open(os.path.join(data_dir, file_name), 'rb') as f_name:
            dic = torch.load(f_name)
        self.rep_type = dic["rep_type"]
        self.rep_dim = dic["rep_dim"]
        self.tuples = []

        bag_rep = dic["bag_rep"]
        self.num_cie = dic["num_cie"]
        self.num_clus = dic["num_clus"]
        self.num_dpt = dic["num_dpt"]
        self.all_tuples = dic["tuples"]

        if bag_type == "cie":
            self.bag_rep = bag_rep[:self.num_cie]
        elif bag_type == "clus":
            self.bag_rep = bag_rep[self.num_cie: self.num_cie + self.num_clus]
        elif bag_type == "dpt":
            self.bag_rep = bag_rep[-self.num_dpt:]
        else:
            raise Exception("Wrong bag type specified: " + bag_type)
        self.select_relevant_tuples(bag_type, self.all_tuples, subsample)

        ##### debug
        # np.random.shuffle(self.tuples)
        # tmp = self.tuples[:10000]
        # self.tuples = tmp

        print("Discriminative Specialized Dataset for split " + split + " loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

    def select_relevant_tuples(self, bag_type, all_tuples, subsample):
        tmp = []
        for person in all_tuples:
            tmp.append({"id": person["id"],
                                "ppl_rep": person["rep"],
                                "bag_rep": self.bag_rep,
                                "label": person[bag_type]
                                })
            if subsample > 0:
                np.random.shuffle(self.tuples)
                self.tuples = tmp[:subsample]
            else:
                self.tuples = tmp

    def get_num_bag(self):
        return len(self.bag_rep)
