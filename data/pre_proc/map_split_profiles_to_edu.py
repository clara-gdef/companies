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
    edu_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu.pkl")
    with open(edu_file, "rb") as f:
        edu = pkl.load(f)

    splits = {}
    for split in ["TRAIN", 'VALID', "TEST"]:
        split_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_" + split + ".pkl")
        with open(split_file, "rb") as f:
            splits[split] = pkl.load(f)
    ipdb.set_trace()




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
