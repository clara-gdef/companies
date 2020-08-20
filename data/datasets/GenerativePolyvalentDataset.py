import os
import pickle as pkl
import ipdb
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class GenerativePolyvalentDataset(Dataset):
    def __init__(self, data_dir, ppl_file, rep_type, cie_reps_file, clus_reps_file, dpt_reps_file, agg_type, load,
                 split="TEST"):
        if load:
            with open(os.path.join(data_dir, "gen_poly_indices.pkl"), 'rb') as f_name:
                dico = torch.load(f_name)
            self.tuples = dico["tuples"]
            self.num_cie = dico["num_cie"]
            self.num_clus = dico["num_clus"]
            self.num_dpt = dico["num_dpt"]
            print("Building Generative Polyvalent Dataset for split " + split + " loaded.")
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
            self.tuples = []

            for person_id in tqdm(ppl_lookup.keys(), desc="Building Generative Polyvalent Dataset for split " + split + " ..."):
                for cie in ppl_reps.keys():
                    for clus in ppl_reps[cie].keys():
                        if len(ppl_reps[cie][clus].keys()) > 0:
                            try:
                                # if person_id in ppl_reps[cie][clus]["id_ppl"]:
                                #         self.tuples.append((ppl_lookup[person_id][rep_type],
                                #                             cie_reps[ppl_lookup[person_id]["cie_label"]][rep_type],
                                #                             1.))
                                #         self.tuples.append((ppl_lookup[person_id][rep_type],
                                #                             clus_reps[ppl_lookup[person_id]["clus_label"]][rep_type],
                                #                             1.))
                                #         self.tuples.append((ppl_lookup[person_id][rep_type],
                                #                             dpt_reps[ppl_lookup[person_id]["dpt_label"]][rep_type],
                                #                             1.))
                                # else:
                                #     if random.random() > 1e-4:
                                #         self.tuples.append((ppl_lookup[person_id][rep_type],
                                #                             cie_reps[ppl_lookup[person_id]["cie_label"]][rep_type],
                                #                             -1.))
                                #         self.tuples.append((ppl_lookup[person_id][rep_type],
                                #                             clus_reps[ppl_lookup[person_id]["clus_label"]][rep_type],
                                #                             -1.))
                                #         self.tuples.append((ppl_lookup[person_id][rep_type],
                                #                             dpt_reps[ppl_lookup[person_id]["dpt_label"]][rep_type],
                                #                             -1.))
                                if person_id in ppl_reps[cie][clus]["id_ppl"]:
                                        self.tuples.append((person_id,
                                                            ppl_lookup[person_id]["cie_label"],
                                                            1.))
                                        self.tuples.append((person_id,
                                                            ppl_lookup[person_id]["clus_label"],
                                                            1.))
                                        self.tuples.append((person_id,
                                                            ppl_lookup[person_id]["dpt_label"],
                                                            1.))
                                else:
                                    if random.random() > 1e-4:
                                        self.tuples.append((person_id,
                                                            ppl_lookup[person_id]["cie_label"],
                                                            -1.))
                                        self.tuples.append((person_id,
                                                            ppl_lookup[person_id]["clus_label"],
                                                            -1.))
                                        self.tuples.append((person_id,
                                                            ppl_lookup[person_id]["dpt_label"],
                                                            -1.))
                            except:
                                continue

            print("Generative Polyvalent Dataset built.")
            print("Dataset Length: " + str(len(self.tuples)))
            self.num_cie = len(cie_reps)
            self.num_clus = len(clus_reps)
            self.num_dpt = len(dpt_reps)
            self.save_dataset(data_dir)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

    def save_dataset(self, data_dir):
        with open(os.path.join(data_dir, "gen_poly_indices.pkl"), 'wb') as f_name:
            dico = {"tuples": self.tuples,
                    'num_cie': self.num_cie,
                    'num_clus': self.num_clus,
                    "num_dpt": self.num_dpt}
            torch.save(dico, f_name)
