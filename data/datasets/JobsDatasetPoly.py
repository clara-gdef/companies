import os
import pickle as pkl
import itertools

import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JobsDatasetPoly(Dataset):
    def __init__(self, data_dir, cie_reps_file, clus_reps_file, dpt_reps_file, ppl_file, load, split, standardized, subsample):
        if load:
            print("Loading previously saved dataset...")
            self.load_dataset(data_dir, split, standardized, subsample)
        else:
            if standardized is True:
                file_name = "total_rep_jobs_unflattened_" + split + "_standardized.pkl"
                with open(os.path.join(data_dir, file_name), 'rb') as f_name:
                    ppl_reps = pkl.load(f_name)
                with open(os.path.join(data_dir, ppl_file + '_' + split + "_standardized.pkl"), 'rb') as f_name:
                    ppl_reps_clus = pkl.load(f_name)
                with open(os.path.join(data_dir, "lookup_ppl.pkl"), 'rb') as f_name:
                    ppl_lookup = pkl.load(f_name)
                with open(os.path.join(data_dir, cie_reps_file + "_standardized.pkl"), "rb") as f_name:
                    cie_reps = pkl.load(f_name)
                with open(os.path.join(data_dir, clus_reps_file + "_standardized.pkl"), "rb") as f_name:
                    clus_reps = pkl.load(f_name)
                with open(os.path.join(data_dir, dpt_reps_file + "_standardized.pkl"), "rb") as f_name:
                    dpt_reps = pkl.load(f_name)
            else:
                file_name = "total_rep_jobs_unflattened_" + split + ".pkl"
                with open(os.path.join(data_dir, file_name), 'rb') as f_name:
                    ppl_reps = pkl.load(f_name)
                with open(os.path.join(data_dir, ppl_file + '_' + split + ".pkl"), 'rb') as f_name:
                    ppl_reps_clus = pkl.load(f_name)
                with open(os.path.join(data_dir, "lookup_ppl.pkl"), 'rb') as f_name:
                    ppl_lookup = pkl.load(f_name)
                with open(os.path.join(data_dir, cie_reps_file + ".pkl"), "rb") as f_name:
                    cie_reps = pkl.load(f_name)
                with open(os.path.join(data_dir, clus_reps_file + ".pkl"), "rb") as f_name:
                    clus_reps = pkl.load(f_name)
                with open(os.path.join(data_dir, dpt_reps_file + ".pkl"), "rb") as f_name:
                    dpt_reps = pkl.load(f_name)
            self.rep_dim = 300
            self.num_cie = len(cie_reps)
            self.num_clus = len(clus_reps)
            self.num_dpt = len(dpt_reps)
            self.bag_rep = self.build_bag_reps(cie_reps, clus_reps, dpt_reps)
            self.tuples = build_ppl_tuples(ppl_reps_clus, ppl_reps, ppl_lookup, self.num_cie, self.num_clus, self.num_dpt, split, standardized, subsample)
            self.save_dataset(data_dir, split, standardized)

        print("Job dataset loaded.")
        print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]["id"], self.tuples[idx]["rep"], self.tuples[idx]["cie"], self.tuples[idx]["clus"], self.tuples[idx]["dpt"], self.bag_rep

    def build_bag_reps(self, cie_reps, clus_reps, dpt_reps):
        tmp = torch.FloatTensor(1, self.rep_dim)
        for bag in itertools.chain(cie_reps.values(), clus_reps.values(), dpt_reps.values()):
            tmp = torch.cat((tmp, bag["ft"]), dim=0)
        return tmp[1:]

    def save_dataset(self, datadir, split, standardized):
        ds_dict = {"rep_dim": self.rep_dim,
                   "num_cie": self.num_cie,
                   "num_clus": self.num_clus,
                   "num_dpt": self.num_dpt,
                   "bag_rep": self.bag_rep,
                   "tuples": self.tuples}
        if standardized is True:
            tgt_file = os.path.join(datadir, "JobsDatasetPoly_" + split + "_standardized.pkl")
            with open(tgt_file, 'wb') as f:
                pkl.dump(ds_dict, f)
        else:
            tgt_file = os.path.join(datadir, "JobsDatasetPoly_" + split + ".pkl")
            with open(tgt_file, 'wb') as f:
                pkl.dump(ds_dict, f)

    def load_dataset(self, datadir, split, standardized, subsample):
        if standardized is True:
            tgt_file = os.path.join(datadir, "JobsDatasetPoly_" + split + "_standardized.pkl")
            with open(tgt_file, 'rb') as f:
                ds_dict = pkl.load(f)
        else:
            tgt_file = os.path.join(datadir, "JobsDatasetPoly_" + split + ".pkl")
            with open(tgt_file, 'rb') as f:
                ds_dict = pkl.load(f)

        self.rep_dim = ds_dict["rep_dim"]
        self.num_cie = ds_dict["num_cie"]
        self.num_clus = ds_dict["num_clus"]
        self.num_dpt = ds_dict["num_dpt"]
        self.bag_rep = ds_dict["bag_rep"]
        if subsample > 0:
            np.random.shuffle(ds_dict["tuples"])
            self.tuples = ds_dict["tuples"][:subsample]
        else:
            self.tuples = ds_dict["tuples"]


def build_ppl_tuples(ppl_reps_clus, ppl_reps, ppl_lookup, num_cie, num_clus, num_dpt, split, standardized, subsample):
    lookup_to_reps = {}
    for cie in ppl_reps.keys():
        lookup_to_reps[cie] = {}
        if standardized and split == "TRAIN":
            for identifier, profile in zip(ppl_reps[cie]["id"], ppl_reps[cie]["profiles"]):
                lookup_to_reps[cie][identifier] = ppl_reps[cie]["profiles"][profile]
        else:
            for identifier, profile in zip(ppl_reps[cie]["id"], ppl_reps[cie]["profiles"]):
                lookup_to_reps[cie][identifier] = profile
    # length of the profile that accounts for 90% of the training dataset
    max_prof_len = 9
    # tmp = []
    # for cie in tqdm(ppl_reps_clus.keys(), desc="Getting mean and std for Discriminative Polyvalent Job Dataset for split " + split + " ..."):
    #     for clus in ppl_reps_clus[cie].keys():
    #         if len(ppl_reps_clus[cie][clus].keys()) > 0:
    #             for person_id in ppl_reps_clus[cie][clus]["id_ppl"]:
    #                 tmp.extend(lookup_to_reps[cie][person_id])
    # ds_mean = np.mean(np.stack(tmp))
    # ds_std = np.std(np.stack(tmp))
    tuples = []
    for cie in tqdm(ppl_reps_clus.keys(), desc="Building Discriminative Polyvalent Job Dataset for split " + split + " ..."):
        for clus in ppl_reps_clus[cie].keys():
            if len(ppl_reps_clus[cie][clus].keys()) > 0:
                for person_id in ppl_reps_clus[cie][clus]["id_ppl"]:
                    assert ppl_lookup[person_id]["cie_label"] <= num_cie - 1
                    assert num_cie <= ppl_lookup[person_id]["clus_label"] <= num_cie + num_clus - 1
                    assert num_cie + num_clus <= ppl_lookup[person_id]["dpt_label"] <= num_cie + num_clus + num_dpt - 1
                    rep = np.zeros((max_prof_len, 300))
                    if not standardized and split != "TRAIN":
                        for num, j in enumerate(lookup_to_reps[cie][person_id]):
                            if num < max_prof_len:
                                #rep[num, :] = (j - ds_mean) / ds_std
                                rep[num, :] = j
                    else:
                        for num, j in enumerate(lookup_to_reps[cie][person_id]):
                            if num < max_prof_len:
                                rep[num, :] = j
                    tuples.append(
                        {"id": person_id,
                         "jobs_len": len(lookup_to_reps[cie][person_id]),
                         "rep": rep,
                         "cie": ppl_lookup[person_id]["cie_label"],
                         "clus": ppl_lookup[person_id]["clus_label"],
                         "dpt": ppl_lookup[person_id]["dpt_label"]
                         }
                    )
    return tuples

