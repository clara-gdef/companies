import argparse
import os
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
        ft_model = fastText.load_model(os.path.join(CFG["modeldir"], args.ft_model))
        print("Word vectors loaded.")
        for item in ["TRAIN", "VALID", "TEST"]:
            data_file = os.path.join(CFG["datadir"], "cie_" + item + ".pkl")
            with open(data_file, "rb") as f:
                data_cie = pkl.load(f)

            ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_" + item + ".pkl")
            with open(ppl_file, 'rb') as fp:
                data_ppl = pkl.load(fp)

            ppl_dict = {}
            for person in data_ppl:
                ppl_dict[person[0]] = {'jobs': person[3], "edu": person[2]}

            with ipdb.launch_ipdb_on_exception():
                cie_dict = dict()
                for cie in tqdm(data_cie.keys(), desc="Processing company..."):
                    cie_dict[cie] = {"id": [], "profiles": [], "edu": []}
                    for year in range(args.year_end, args.year_start, -1):
                        if year in data_cie[cie].keys():
                            for person in data_cie[cie][year]:
                                person_id = person[0]
                                if person_id not in cie_dict[cie]["id"]:
                                    jobs = ppl_dict[person_id]["jobs"]
                                    edu = ppl_dict[person_id]["edu"]
                                    cie_dict[cie]["id"].append(person_id)
                                    cie_dict[cie]["profiles"].append(to_emb(jobs, ft_model, args.flat))
                                    # cie_dict[cie]["edu"].append(to_emb(edu, ft_model, args.flat))
            avg_len = [len(cie_dict[cie]["id"]) for cie in cie_dict.keys()]
            print("avg num of ppl per cie: " + str(int(np.mean(np.asarray(avg_len)))))
            target = os.path.join(CFG["datadir"], args.tgt_file + item + ".pkl")
            with open(target, "wb") as f:
                pkl.dump(cie_dict, f)


def to_emb(complete_profile, ft_model, flat):
    word_count = 0
    if flat:
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
    else:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_file", type=str, default="total_rep_jobs_")
    parser.add_argument("--ft_model", type=str, default='ft_fs_edu_job.bin')
    parser.add_argument("--min_count_ppl", type=int, default=200)
    parser.add_argument("--year_start", type=int, default=2000)
    parser.add_argument("--year_end", type=int, default=2017)
    parser.add_argument("--flat", type=bool, default=False)
    args = parser.parse_args()
    main(args)
