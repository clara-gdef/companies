import os
import pickle as pkl
import ipdb
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class GenerativePolyvalentDataset(Dataset):
    def __init__(self, data_dir, ppl_file, cie_reps_file, clus_reps_file, dpt_reps_file, rep_type, load,
                 split="TEST"):
        if load:
            with open(os.path.join(data_dir, "gen_poly_indices_" + split + ".pkl"), 'rb') as f_name:
                dico = torch.load(f_name)
            self.tuples = dico["tuples"]
            self.rep_type = dico["rep_type"]
            self.cie_reps = dico["cie_reps"]
            self.clus_reps = dico["clus_reps"]
            self.dpt_reps = dico["dpt_reps"]
            self.ppl_lookup = dico["ppl_lookup"]
            self.bags_reps = {**self.cie_reps, **self.clus_reps, **self.dpt_reps}
            ipdb.set_trace()

            print("Generative Polyvalent Dataset for split " + split + " loaded.")
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
                                    if random.random() < 1e-4:
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
            self.cie_reps = cie_reps
            self.clus_reps = clus_reps
            self.dpt_reps = dpt_reps
            self.rep_type = rep_type
            self.ppl_lookup = ppl_lookup
            self.bags_reps = {**self.cie_reps, **self.clus_reps, **self.dpt_reps}
            self.save_dataset(data_dir, split)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return (self.ppl_lookup[self.tuples[idx][0]][self.rep_type],
                self.bags_reps[self.tuples[idx][1]],
                self.tuples[idx][-1]
                )

    def save_dataset(self, data_dir, split):
        dico = {"tuples": self.tuples,
                "rep_type": self.rep_type,
                "ppl_lookup": self.ppl_lookup,
                'cie_reps': self.cie_reps,
                'clus_reps': self.clus_reps,
                "dpt_reps": self.dpt_reps}
        with open(os.path.join(data_dir, "gen_poly_indices_" + split + ".pkl"), 'wb') as f_name:
            torch.save(dico, f_name)
