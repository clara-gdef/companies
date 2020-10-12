import os
import argparse
import pickle as pkl
import ipdb
import yaml
from tqdm import tqdm
import re
from collections import Counter
import pandas as pd
import unidecode
from nltk.tokenize import word_tokenize
import json
from datetime import datetime


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu")
    tgt_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_delta_t")
    for split in ["_TRAIN", "_VALID", "_TEST"]:
        with open(ppl_file + split + ".pkl", 'rb') as fp:
            data = pkl.load(fp)
        dataset = build_dataset(data)
        with open(tgt_file + split + ".pkl", 'wb') as f:
            pkl.dump(dataset)


def build_dataset(data, split):
    all_durations_days = []
    faulty_profiles = set()
    good_profiles = set()
    faulty_jobs = []
    total_jobs = 0
    for person in tqdm(data):
        flag = False
        for job in person[-1]:
            total_jobs += 1
            delta = (datetime.fromtimestamp(job["to"]) - datetime.fromtimestamp(job["from"])).days
            if delta < 0:
                flag = True
                faulty_profiles.add(person[0])
                faulty_jobs.append(job)
            else:
                all_durations_days.append(delta)
        if not flag:
            good_profiles.add(person)
    final_dataset = []
    print("percentage of wrongly filled jobs for split " + split +  ": " + str(100 * len(faulty_jobs)/total_jobs) + "%")
    print("percentage of removed profiles for split " + split +  ": " + str(100 * len(faulty_profiles)/len(data)) + "%")
    for person in good_profiles:
        new_p = add_deltas_to_jobs(person)
        final_dataset.append(new_p)


def add_deltas_to_jobs(person):
    for job in person[-1]:
        delta = (datetime.fromtimestamp(job["to"]) - datetime.fromtimestamp(job["from"])).days
        job["delta_days"] = delta
    return person


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="bp_3jobs_desc_edu_skills_industry_date_company_FR.json")
    parser.add_argument("--index_file", type=str, default='pkl/index_40k.pkl')
    parser.add_argument("--temporal", type=bool, default=False)
    parser.add_argument("--MAX_SEQ_LENGTH", type=int, default=64)
    parser.add_argument("--DATA_DIR_SRC", type=str, default="/local/gainondefor/work/lip6/data/seq2seq")
    parser.add_argument("--DATA_DIR_TGT", type=str, default="/local/gainondefor/work/lip6/data/companies")
    parser.add_argument("--MIN_JOB_COUNT", type=int, default=3)
    args = parser.parse_args()
    main(args)
