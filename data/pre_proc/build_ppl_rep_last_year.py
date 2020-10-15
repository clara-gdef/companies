import argparse
import os
import torch
import pickle as pkl

import fastText
import yaml
from tqdm import tqdm
import ipdb
import numpy as np

def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        print("Loading word vectors...")
        if args.edu:
            ft_model = fastText.load_model(os.path.join(CFG["modeldir"], "ft_fs_edu_job.bin"))
        else:
            ft_model = fastText.load_model(os.path.join(CFG["modeldir"], "ft_fs.bin"))
        print("Word vectors loaded.")

        ds_mean, ds_std = build_for_train(ft_model)

        for item in ["VALID", "TEST"]:
            data_file = os.path.join(CFG["datadir"], "cie_" + item + ".pkl")
            with open(data_file, "rb") as f:
                data_cie = pkl.load(f)

            if args.edu:
                ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_" + item + ".pkl")
                with open(ppl_file, 'rb') as fp:
                    data_ppl = pkl.load(fp)
            else:
                ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_" + item + ".pkl")
                with open(ppl_file, 'rb') as fp:
                    data_ppl = pkl.load(fp)

            ppl_dict = {}
            for person in data_ppl:
                if args.edu:
                    ppl_dict[person[0]] = {'jobs': person[3], "edu": person[2]}
                else:
                    ppl_dict[person[0]] = {'jobs': person[3]}

            with ipdb.launch_ipdb_on_exception():
                cie_dict = dict()
                for cie in tqdm(data_cie.keys(), desc="Processing company..."):
                    if args.edu:
                        cie_dict[cie] = {"id": [], "profiles": [], "edu": []}
                    else:
                        cie_dict[cie] = {"id": [], "profiles": []}
                    for year in range(args.year_end, args.year_start, -1):
                        if year in data_cie[cie].keys():
                            for person in data_cie[cie][year]:
                                person_id = person[0]
                                if person_id not in cie_dict[cie]["id"]:
                                    jobs = ppl_dict[person_id]["jobs"]
                                    cie_dict[cie]["id"].append(person_id)
                                    tmp = to_emb(jobs, ft_model, args.flat)
                                    if tmp is None:
                                        ipdb.set_trace()
                                    cie_dict[cie]["profiles"].append((tmp - ds_mean) / ds_std)
                                    if args.edu:
                                        edu = ppl_dict[person_id]["edu"]
                                        cie_dict[cie]["edu"].append(to_emb(edu, ft_model, args.flat))
            avg_len = [len(cie_dict[cie]["id"]) for cie in cie_dict.keys()]
            print("avg num of ppl per cie: " + str(int(np.mean(np.asarray(avg_len)))))
            f_name = args.tgt_file
            if args.flat == "True":
                f_name += "unflattened_"
            if args.edu:
                f_name += "edu_"
            target = os.path.join(CFG["datadir"], f_name + item + "_standardized.pkl")
            with open(target, "wb") as f:
                pkl.dump(cie_dict, f)


def build_for_train(ft_model):
    data_file = os.path.join(CFG["datadir"], "cie_TRAIN.pkl")
    with open(data_file, "rb") as f:
        data_cie = pkl.load(f)
    if args.edu:
        ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_TRAIN.pkl")
        with open(ppl_file, 'rb') as fp:
            data_ppl = pkl.load(fp)
    else:
        ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_TRAIN.pkl")
        with open(ppl_file, 'rb') as fp:
            data_ppl = pkl.load(fp)

    ppl_dict = {}
    for person in data_ppl:
        if args.edu:
            ppl_dict[person[0]] = {'jobs': person[3], "edu": person[2]}
        else:
            ppl_dict[person[0]] = {'jobs': person[3]}

    with ipdb.launch_ipdb_on_exception():
        cie_dict = dict()
        for cie in tqdm(data_cie.keys(), desc="Processing company..."):
            if args.edu:
                cie_dict[cie] = {"id": [], "profiles": [], "edu": []}
            else:
                cie_dict[cie] = {"id": [], "profiles": []}
            for year in range(args.year_end, args.year_start, -1):
                if year in data_cie[cie].keys():
                    for person in data_cie[cie][year]:
                        person_id = person[0]
                        if person_id not in cie_dict[cie]["id"]:
                            jobs = ppl_dict[person_id]["jobs"]
                            cie_dict[cie]["id"].append(person_id)
                            tmp = to_emb(jobs, ft_model, args.flat)
                            if tmp is None:
                                ipdb.set_trace()
                            cie_dict[cie]["profiles"].append(tmp)
                            if args.edu:
                                edu = ppl_dict[person_id]["edu"]
                                cie_dict[cie]["edu"].append(to_emb(edu, ft_model, args.flat))

    stacked_ppl = np.zeros(300)
    for cie in tqdm(cie_dict.keys(), desc="Finding mean and std per dimenson across the dataset..."):
        if args.flat == "True":
            ipdb.set_trace()
        else:
            for profile in cie_dict[cie]["profiles"]:
                stacked_ppl = np.concatenate((stacked_ppl, profile), axis=0)
    tmp = stacked_ppl[1:]

    ds_mean = np.mean(tmp, axis=0)
    ds_std = np.std(tmp, axis=0)

    cie_dict_standardized = dict()
    for cie in tqdm(cie_dict.keys(), desc="Processing company for standardization..."):
        cie_dict_standardized[cie] = {"profiles": []}
        cie_dict_standardized[cie]["id"] = cie_dict[cie]["id"]
        if args.edu:
            cie_dict_standardized[cie]["edu"] = cie_dict[cie]["edu"]
        if args.flat == "True":
            ipdb.set_trace()
        else:
            for profile in cie_dict[cie]["profiles"]:
                cie_dict_standardized[cie]["profiles"].append((profile - ds_mean)/ds_std)
    avg_len = [len(cie_dict[cie]["id"]) for cie in cie_dict.keys()]
    print("avg num of ppl per cie: " + str(int(np.mean(np.asarray(avg_len)))))
    f_name = args.tgt_file
    if args.flat:
        f_name += "unflattened_"
    if args.edu:
        f_name += "edu_"
    target = os.path.join(CFG["datadir"], f_name + "TRAIN_standardized.pkl")
    with open(target, "wb") as f:
        pkl.dump(cie_dict_standardized, f)

    return ds_mean, ds_std


def to_emb(complete_profile, ft_model, flat):
    word_count = 0
    if flat == "True":
        embs = np.zeros((len(complete_profile), ft_model.get_dimension()))
        for num, job in enumerate(complete_profile):
            tmp = []
            if type(job) == dict:
                for token in job["job"]:
                    tmp.append(ft_model.get_word_vector(token))
                    word_count += 1
            else:
                for token in job:
                    tmp.append(ft_model.get_word_vector(token))
                    word_count += 1
            embs[num, :] = np.mean(np.stack(tmp), axis=0) / word_count
        return embs
    else:
        emb = np.zeros((ft_model.get_dimension()))
        for job in complete_profile:
            if type(job) == dict:
                for token in job["job"]:
                    word_count += 1
                    emb += ft_model.get_word_vector(token)
            else:
                for token in job:
                    word_count += 1
                    emb += ft_model.get_word_vector(token)
        return emb / word_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_file", type=str, default="total_rep_jobs_")
    parser.add_argument("--min_count_ppl", type=int, default=200)
    parser.add_argument("--year_start", type=int, default=2000)
    parser.add_argument("--year_end", type=int, default=2017)
    parser.add_argument("--flat", default=False)
    parser.add_argument("--edu", type=bool, default=False)
    args = parser.parse_args()
    main(args)
