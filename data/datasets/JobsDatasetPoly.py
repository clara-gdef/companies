import os
import pickle as pkl
import itertools

import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDatasetPoly(Dataset):
    def __init__(self, data_dir, cie_reps_file, clus_reps_file, dpt_reps_file, ppl_file, load, split):
        self.jobs = []
        self.jobs_emb = []
        self.indices = []
        self.preds = []
        self.labels = []
        print("Loading previously saved dataset...")
        if load:
            ipdb.set_trace()
        else:
            file_name = "total_rep_jobs_unflattened_" + split + ".pkl"
            with open(os.path.join(data_dir, file_name), 'rb') as f_name:
                ppl_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, ppl_file + '_' + split + ".pkl"), 'rb') as f_name:
                ppl_reps_clus = pkl.load(f_name)
            with open(os.path.join(data_dir, "lookup_ppl.pkl"), 'rb') as f_name:
                ppl_lookup = pkl.load(f_name)
            with open(os.path.join(data_dir, cie_reps_file), "rb") as f_name:
                cie_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, clus_reps_file), "rb") as f_name:
                clus_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, dpt_reps_file), "rb") as f_name:
                dpt_reps = pkl.load(f_name)
            self.rep_dim = 300
            self.num_cie = len(cie_reps)
            self.num_clus = len(clus_reps)
            self.num_dpt = len(dpt_reps)
            self.bag_rep = self.build_bag_reps(cie_reps, clus_reps, dpt_reps)
            self.tuples = build_ppl_tuples(ppl_reps_clus, ppl_reps, ppl_lookup, self.num_cie, self.num_clus, self.num_dpt, split)


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
        return

    def __getitem__(self, idx):
        return

    def build_bag_reps(self, cie_reps, clus_reps, dpt_reps):
        tmp = torch.FloatTensor(1, self.rep_dim)
        for bag in itertools.chain(cie_reps.values(), clus_reps.values(), dpt_reps.values()):
            tmp = torch.cat((tmp, bag["ft"]), dim=0)
        return tmp[1:]


def build_ppl_tuples(ppl_reps_clus, ppl_reps, ppl_lookup, num_cie, num_clus, num_dpt, split):
    lookup_to_reps = {}
    for cie in ppl_reps.keys():
        lookup_to_reps[cie] = {}
        for identifier, profile in zip(ppl_reps[cie]["id"], ppl_reps[cie]["profiles"]):
            lookup_to_reps[cie][identifier] = profile
    tmp = []
    for cie in tqdm(ppl_reps_clus.keys(), desc="Getting mean and std for Discriminative Polyvalent Job Dataset for split " + split + " ..."):
        for clus in ppl_reps_clus[cie].keys():
            if len(ppl_reps_clus[cie][clus].keys()) > 0:
                for person_id in ppl_reps_clus[cie][clus]["id_ppl"]:
                    tmp.extend(lookup_to_reps[cie][person_id])
    ds_mean = torch.mean(torch.stack(tmp))
    ds_std = torch.std(torch.stack(tmp))
    tuples = []
    for cie in tqdm(ppl_reps_clus.keys(), desc="Building Discriminative Polyvalent Job Dataset for split " + split + " ..."):
        for clus in ppl_reps_clus[cie].keys():
            if len(ppl_reps_clus[cie][clus].keys()) > 0:
                for person_id in ppl_reps_clus[cie][clus]["id_ppl"]:
                    assert ppl_lookup[person_id]["cie_label"] <= num_cie - 1
                    assert num_cie <= ppl_lookup[person_id]["clus_label"] <= num_cie + num_clus - 1
                    assert num_cie + num_clus <= ppl_lookup[person_id]["dpt_label"] <= num_cie + num_clus + num_dpt - 1
                    tuples.append(
                        {"id": person_id,
                         "rep": (ppl_reps[person_id]["profiles"] - ds_mean) / ds_std,
                         "cie": ppl_lookup[person_id]["cie_label"],
                         "clus": ppl_lookup[person_id]["clus_label"],
                         "dpt": ppl_lookup[person_id]["dpt_label"]
                         }
                    )
    return tuples
