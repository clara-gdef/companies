import os
import pickle as pkl
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativeSpecializedDataset(Dataset):
    def __init__(self, data_dir, rep_type, agg_type, bag_type, split):
        print("Loading previously saved dataset...")
        file_name = "disc_poly_" + agg_type + "_" + rep_type + "_" + split + ".pkl"
        with open(os.path.join(data_dir, file_name), 'rb') as f_name:
            dic = torch.load(f_name)
        self.rep_type = dic["rep_type"]
        self.rep_dim = dic["rep_dim"]
        self.tuples = []

        bag_rep = dic["bag_rep"]
        num_cie = dic["num_cie"]
        num_clus = dic["num_clus"]
        num_dpt = dic["num_dpt"]
        all_tuples = dic["tuples"]
        if bag_type == "cie":
            self.bag_rep = bag_rep[:num_cie]
        elif bag_type == "clus":
            self.bag_rep = bag_rep[num_cie: num_cie + num_clus]
        elif bag_type == "dpt":
            self.bag_rep = bag_rep[-num_dpt:]
        else:
            raise Exception("Wrong bag type specified: " + bag_type)
        self.select_relevant_tuples(bag_type, all_tuples)

        print("Discriminative Specialized Dataset for split " + split + " loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx], self.bag_rep

    def select_relevant_tuples(self, bag_type, all_tuples):
        for person in all_tuples:
            self.tuples.append({"id": person["id"],
                                "rep": person["rep"],
                                bag_type: person[bag_type]
                                })
