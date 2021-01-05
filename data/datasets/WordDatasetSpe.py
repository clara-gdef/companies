import os
import pickle as pkl
import ipdb
import itertools
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class WordDatasetSpe(Dataset):
    def __init__(self, data_dir, rep_type, embedder, split, subsample, load, min_job_count, max_job_count, max_word_count=126):
        self.data_dir = data_dir
        self.split = split
        self.rep_type = rep_type
        if load:
            self.load_dataset()
        else:
            self.MIN_JOB_COUNT = min_job_count
            self.MAX_JOB_COUNT = max_job_count
            self.MAX_WORD_COUNT = max_word_count

            if rep_type == "ft":
                self.embedder_dim = 300
            else:
                self.embedder_dim = 1024
            if self.split == "TRAIN":
                ds_mean, ds_std = None, None
            else:
                with open(os.path.join(data_dir, "word_dataset_params.pkl"), 'rb') as f_name:
                    tmp = pkl.load(f_name)
                ds_mean, ds_std = tmp[0], tmp[1]
            self.tuples = []

            with open(os.path.join(data_dir, "lookup_ppl.pkl"), 'rb') as f_name:
                lookup = pkl.load(f_name)

            split_ids = self.get_split_indices(split)
            # TODO remove slicing
            self.build_profiles_from_indices(split_ids[:100], embedder, lookup, ds_mean, ds_std)

            if subsample > 0:
                np.random.shuffle(self.tuples)
                tmp = self.tuples[:subsample]
                self.tuples = tmp

            print("Word dataset Dataset for split " + split + " built.")
            print("Dataset Length: " + str(len(self.tuples)))
            self.save_dataset()


    def get_split_indices(self, split):
        file_name = "total_rep_jobs_unflattened_" + split + ".pkl"
        with open(os.path.join(self.data_dir, file_name), 'rb') as f_name:
            ppl_reps = pkl.load(f_name)
        ids = []
        for cie in tqdm(ppl_reps.keys()):
            ids.extend(ppl_reps[cie]["id"])
        return ids

    def build_profiles_from_indices(self, indices, embedder, lookup,  ds_mean, ds_std):
        base_file = os.path.join(self.data_dir,  "profiles_jobs_skills_edu.pkl")
        with open(base_file, 'rb') as f:
            ppl = pkl.load(f)
        for person in tqdm(ppl):
            if person[0] in indices:
                new_p = self.build_new_person(person, embedder, lookup)
                self.tuples.append(new_p)
        # standardize word embeddings
        tups = self.tuples
        prof_emb_tmp = np.stack([i[-2] for i in self.tuples])
        if ds_std is None and ds_mean is None:
            ipdb.set_trace()
            ds_mean = np.mean(prof_emb_tmp, axis=0)
            ds_std = np.std(prof_emb_tmp, axis=0)
            print("Mean = " + str(ds_mean) + " and STD  = " + str(ds_std) + " computed for train split.")
            with open(os.path.join(self.data_dir, "word_dataset_params.pkl"), 'wb') as f_name:
                pkl.dump([ds_mean, ds_std], f_name)
        ipdb.set_trace()
        self.tuples = []
        for tup in tups:
            jobs_embs = np.zeros((self.MAX_JOB_COUNT, self.MAX_WORD_COUNT, self.embedder_dim))
            for num_job, job in enumerate(tup[2]):
                if num_job < self.MAX_JOB_COUNT:
                    for place, word in enumerate(job):
                        if place < self.MAX_WORD_COUNT:
                            jobs_embs[num_job, place, :] = (tup[-2] - ds_mean)/ds_std
            tup[-2] = jobs_embs
            ipdb.set_trace()
            self.tuples.append(tup)


    def build_new_person(self, person, embedder, lookup):
        new_p = [person[0], len(person[-1])]
        job_words = []
        for job in person[-1]:
            job_words.append(job["job"][:min(self.MAX_WORD_COUNT, len(job["job"]))])
        new_p.append(job_words)
        # jobs_embs = np.zeros((self.MAX_JOB_COUNT, self.MAX_WORD_COUNT, self.embedder_dim))
        jobs_embs = []
        # for num_job, job in enumerate(job_words):
        #     if num_job < self.MAX_JOB_COUNT:
        for job in job_words:
            job_emb = np.zeros((self.MAX_WORD_COUNT, self.embedder_dim))
            for place, word in enumerate(job):
                if place < self.MAX_WORD_COUNT:
                    tmp = embedder(word)
                    if self.rep_type == "ft":
                        job_emb[place, :] = tmp
                    else:
                        ipdb.set_trace()
            jobs_embs.append(job_emb)
        new_p.append(jobs_embs)
        new_p.append(lookup[person[0]]["cie_label"])
        return new_p

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

    def save_dataset(self):
        dico = {}
        for attribute in vars(self):
            if not str(attribute).startswith("__"):
                dico[str(attribute)] = vars(self)[attribute]
        tgt_file = os.path.join(self.datadir, "word_ds_" + self.rep_type + "_" + self.split + '.pkl')
        with open(tgt_file, 'wb') as f:
            pkl.dump(dico, f)
        print("Dataset saved : " + tgt_file)

    def load_dataset(self):
        tgt_file = os.path.join(self.datadir, "word_ds_" + self.rep_type + "_" + self.split + '.pkl')
        with open(tgt_file, 'rb') as f:
            dico = pkl.load(f)
        for key in tqdm(dico, desc="Loading attributes from save..."):
            vars(self)[key] = dico[key]
        print("Dataset load from : " + tgt_file)