import os
import pickle as pkl
import ipdb
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class DiscriminativePolyvalentDataset(Dataset):
    def __init__(self, data_dir, ppl_file, rep_type, cie_reps_file, clus_reps_file, dpt_reps_file, agg_type,
                 load=False):
        if load:
            with open(os.path.join(data_dir, "disc_poly_" + agg_type + "_" + rep_type + ".pkl"), 'rb') as f_name:
                self = torch.load(f_name)

        else:
            print("Loading data...")
            with open(os.path.join(data_dir, "lookup_ppl.pkl"), 'rb') as f_name:
                ppl_lookup = pkl.load(f_name)
            with open(os.path.join(data_dir, ppl_file), 'rb') as f_name:
                ppl_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, cie_reps_file), "rb") as f_name:
                cie_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, clus_reps_file), "rb") as f_name:
                clus_reps = pkl.load(f_name)
            with open(os.path.join(data_dir, dpt_reps_file), "rb") as f_name:
                dpt_reps = pkl.load(f_name)
            print("Data Loaded.")
            self.rep_type = rep_type
            self.tuples = self.build_ppl_tuples(ppl_reps, ppl_lookup, rep_type)
            self.rep_dim = self.tuples[0]["rep"].shape[-1]
            self.bag_rep = self.build_bag_reps(cie_reps, clus_reps, dpt_reps)
            self.save_dataset(data_dir, agg_type, rep_type)

            print("Discriminative Polyvalent Dataset built.")
            print("Dataset Length: " + str(len(self.tuples)))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

    def save_dataset(self, data_dir, agg_type, rep_type):
        with open(os.path.join(data_dir, "disc_poly_" + agg_type + "_" + rep_type + ".pkl"), 'wb') as f_name:
            torch.save(self, f_name)

    def build_ppl_tuples(self, ppl_reps, ppl_lookup, rep_type):
        tuples = []
        for cie in tqdm(ppl_reps.keys(), desc="Building Discriminative Polyvalent Dataset..."):
            for clus in ppl_reps[cie].keys():
                for person_id in ppl_reps[cie][clus]["id_ppl"]:
                    tuples.append(
                        {"id": person_id,
                         "rep": ppl_lookup[person_id][rep_type],
                         "cie": ppl_lookup[person_id]["cie_label"],
                         "clus": ppl_lookup[person_id]["clus_label"],
                         "label": ppl_lookup[person_id]["dpt_label"]
                         }
                    )
        return tuples

    def build_bag_reps(self, cie_reps, clus_reps, dpt_reps):
        tmp = torch.FloatTensor(1, self.rep_dim)
        for bag in itertools.chain(cie_reps.values(), clus_reps.values(), dpt_reps.values()):
            tmp = torch.cat((tmp, bag[self.rep_type]), dim=0)
        return tmp[1:]
