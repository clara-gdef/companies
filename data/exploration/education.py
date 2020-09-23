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


base_file = "bp_3jobs_desc_edu_skills_industry_date_company_FR.json"
MIN_JOB_COUNT = 3
MAX_SEQ_LENGTH = 64


global CFG
with open("../../config.yaml", "r") as ymlfile:
    CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
# with ipdb.launch_ipdb_on_exception():
cie_file = os.path.join(CFG["datadir"], "cie_list.pkl")
with open(cie_file, "rb") as f:
    cie_list = pkl.load(f)
synonym_file = os.path.join(CFG["datadir"], "cie_synonyms.pkl")
with open(synonym_file, "rb") as f:
    syn_cie = pkl.load(f)
blacklist_file = os.path.join(CFG["datadir"], "blacklist.pkl")
with open(blacklist_file, "rb") as f:
    blacklist = pkl.load(f)


def word_seq_into_list(position, description, cie_list, syn_cie):
    number_regex = re.compile(r'\d+(,\d+)?')
    whole_job = position.lower() + ' ' + description.lower()
    new_tup = []

    for cie in cie_list:
        if cie in whole_job.lower():
            if cie in syn_cie.keys():
                handle = syn_cie[cie]
            else:
                handle = cie
            whole_job = whole_job.replace(cie, handle)

    for name in syn_cie.keys():
        if name in whole_job.lower():
            handle = syn_cie[name]
            whole_job = whole_job.replace(cie, handle)

    job = word_tokenize(whole_job)

    for tok in job:
        if re.match(number_regex, tok):
            new_tup.append("NUM")
        elif tok.lower() in cie_list or tok.lower() in syn_cie.keys():
            new_tup.append("CIE")
        else:
            new_tup.append(tok.lower())
    cleaned_tup = [item for item in new_tup if item != ""]
    return cleaned_tup


def handle_date(job):
    if job["to"] == "Present":
        date_time_str = '2018-04-12'  # date of files creation
        time = datetime.timestamp(datetime.strptime(date_time_str, '%Y-%m-%d'))
    elif len(job["to"].split(" ")) == 2:
        try:
            time = datetime.timestamp(datetime.strptime(job["to"], "%B %Y"))
        except ValueError:
            time = datetime.timestamp(datetime.strptime(job["to"].split(" ")[-1], "%Y"))
    else:
        try:
            time = datetime.timestamp(datetime.strptime(job["to"].split(" ")[-1], "%Y"))
        except ValueError:
            date_time_str = '2018-04-13'  # date of files creation
            time = datetime.timestamp(datetime.strptime(date_time_str, '%Y-%m-%d'))
    tstmp = pd.Timestamp.fromtimestamp(time)
    return round(datetime.timestamp(tstmp.round("D").to_pydatetime()))


def get_edu_info(person, cie_list, syn_cie, blacklist):
    education = person[-2]
    jobs = []
    flag = False
    for job in person[1]:
        if 'company' in job.keys():
            threshold = min(len(job["company"].split(" ")), 5)
            tmp = job["company"].split(" ")[:threshold]
            normalized_name = [unidecode.unidecode(name.lower()) for name in tmp]
            company_name = "".join(normalized_name)
            if company_name in cie_list:
                flag = True
    if flag:
        for job in person[1]:
            if 'company' in job.keys():
                threshold = min(len(job["company"].split(" ")), 5)
                tmp = job["company"].split(" ")[:threshold]
                normalized_name = [unidecode.unidecode(name.lower()) for name in tmp]
                company_name = "".join(normalized_name)
                if company_name not in blacklist:
                    end = handle_date(job)
                    tstmp = pd.Timestamp.fromtimestamp(job["from_ts"])
                    start = round(datetime.timestamp(tstmp.round("D").to_pydatetime()))
                    if company_name in syn_cie.keys():
                        cie = syn_cie[company_name]
                    else:
                        cie = company_name
                    if (end > 0) and (start > 0):  # corresponds to the timestamp of 01/01/1970
                        j = {'from': start,
                             'to': end,
                             'company': cie,
                             'job': word_seq_into_list(job["position"],
                                                       job["description"], cie_list, syn_cie)}
                        jobs.append(j)

    return education, jobs



current_file = os.path.join(CFG["prevdatadir"], base_file)
with open(current_file, 'r') as f:
    num_lines = sum(1 for line in f)
with open(current_file, 'r') as f:
    pbar = tqdm(f, total=num_lines)
    edu_backgrounds = []
    for line in pbar:
        try:
            current_person = json.loads(line)
            jobs = current_person[1]
            skills = current_person[2]
            if len(jobs) >= MIN_JOB_COUNT and len(skills) > 0:
                edu_info, new_jobs = get_edu_info(current_person, cie_list, syn_cie, blacklist)
                if len(new_jobs) >= MIN_JOB_COUNT:
                    edu_backgrounds.extend(edu_info)
        except:
            continue
        pbar.update(1)
tgt_file = "unprocessed_educations.pkl"
with open(os.path.join(CFG["datadir"], tgt_file), "wb") as f:
    pkl.dump(edu_backgrounds, f)
print("File created!")
