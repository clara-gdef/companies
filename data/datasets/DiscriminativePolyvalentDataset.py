import os
import pickle as pkl
import itertools
import torch
import numpy as np
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativePolyvalentDataset(Dataset):
    def __init__(self, data_dir,
                 ppl_file, rep_type,
                 cie_reps_file, clus_reps_file,
                 dpt_reps_file, agg_type,
                 split,
                 load):
        if load:
            print("Loading previously saved dataset...")
            file_name = "disc_poly_" + agg_type + "_" + rep_type + "_" + split + ".pkl"
            with open(os.path.join(data_dir, file_name), 'rb') as f_name:
                dic = torch.load(f_name)
            self.rep_type = dic["rep_type"]
            r = torch.randperm(14000)
            self.tuples = dic["tuples"][:r]
            self.rep_dim = dic["rep_dim"]
            self.bag_rep = dic["bag_rep"]
            self.num_cie = dic["num_cie"]
            self.num_clus = dic["num_clus"]
            self.num_dpt = dic["num_dpt"]
            print("Discriminative Polyvalent Dataset for split " + split + " loaded.")
            print("Dataset Length: " + str(len(self.tuples)))

        else:
            print("Loading data...")
            with open(os.path.join(data_dir, "lookup_ppl.pkl"), 'rb') as f_name:
                ppl_lookup = pkl.load(f_name)
            with open(os.path.join(data_dir, ppl_file + '_' + split + ".pkl"), 'rb') as f_name:
                ppl_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, cie_reps_file), "rb") as f_name:
                cie_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, clus_reps_file), "rb") as f_name:
                clus_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, dpt_reps_file), "rb") as f_name:
                dpt_reps = pkl.load(f_name)
            print("Data Loaded.")
            self.rep_type = rep_type
            self.num_cie = len(cie_reps)
            self.num_clus = len(clus_reps)
            self.num_dpt = len(dpt_reps)
            self.tuples = build_ppl_tuples(ppl_reps, ppl_lookup, rep_type, self.num_cie, self.num_clus, self.num_dpt, split)
            self.rep_dim = self.tuples[0]["rep"].shape[-1]
            self.bag_rep = self.build_bag_reps(cie_reps, clus_reps, dpt_reps)
            self.save_dataset(data_dir, agg_type, rep_type, split)

            print("Discriminative Polyvalent Dataset for split " + split + " built.")
            print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx], self.bag_rep

    def save_dataset(self, data_dir, agg_type, rep_type, split):
        dictionary = {"rep_type": self.rep_type,
                      "tuples": self.tuples,
                      "rep_dim": self.rep_dim,
                      "bag_rep": self.bag_rep,
                      'num_cie': self.num_cie,
                      'num_clus': self.num_clus,
                      "num_dpt": self.num_dpt}
        file_name = "disc_poly_" + agg_type + "_" + rep_type + "_" + split + ".pkl"
        with open(os.path.join(data_dir, file_name), 'wb') as f_name:
            torch.save(dictionary, f_name)

    def build_bag_reps(self, cie_reps, clus_reps, dpt_reps):
        tmp = torch.FloatTensor(1, self.rep_dim)
        for bag in itertools.chain(cie_reps.values(), clus_reps.values(), dpt_reps.values()):
            tmp = torch.cat((tmp, bag[self.rep_type]), dim=0)
        return tmp[1:]


def build_ppl_tuples(ppl_reps, ppl_lookup, rep_type, num_cie, num_clus, num_dpt, split):
    tmp = []
    for cie in tqdm(ppl_reps.keys(), desc="Getting mean and std for Discriminative Polyvalent Dataset for split " + split + " ..."):
        for clus in ppl_reps[cie].keys():
            if len(ppl_reps[cie][clus].keys()) > 0:
                for person_id in ppl_reps[cie][clus]["id_ppl"]:
                    tmp.append(ppl_lookup[person_id][rep_type])
    ds_mean = torch.mean(torch.stack(tmp))
    ds_std = torch.std(torch.stack(tmp))
    tuples = []
    for cie in tqdm(ppl_reps.keys(), desc="Building Discriminative Polyvalent Dataset for split " + split + " ..."):
        for clus in ppl_reps[cie].keys():
            if len(ppl_reps[cie][clus].keys()) > 0:
                for person_id in ppl_reps[cie][clus]["id_ppl"]:
                    assert ppl_lookup[person_id]["cie_label"] <= num_cie - 1
                    assert num_cie <= ppl_lookup[person_id]["clus_label"] <= num_cie + num_clus - 1
                    assert num_cie + num_clus <= ppl_lookup[person_id]["dpt_label"] <= num_cie + num_clus + num_dpt - 1
                    tuples.append(
                        {"id": person_id,
                         "rep": (ppl_lookup[person_id][rep_type] - ds_mean) / ds_std,
                         "cie": ppl_lookup[person_id]["cie_label"],
                         "clus": ppl_lookup[person_id]["clus_label"],
                         "dpt": ppl_lookup[person_id]["dpt_label"]
                         }
                    )
    return tuples
