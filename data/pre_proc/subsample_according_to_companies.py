import os
import argparse
import pickle as pkl
import ipdb
from tqdm import tqdm
import re
from collections import Counter
import pandas as pd
import unidecode
import nltk
from nltk.tokenize import word_tokenize
import json
from datetime import datetime
from utils.Utils import getDuration


def main(args):
    # with ipdb.launch_ipdb_on_exception():
    cie_file = os.path.join(args.DATA_DIR_TGT, "cie_list.pkl")
    with open(cie_file, "rb") as f:
        cie_list = pkl.load(f)
    synonym_file = os.path.join(args.DATA_DIR_TGT, "cie_synonyms.pkl")
    with open(synonym_file, "rb") as f:
        syn_cie = pkl.load(f)
    blacklist_file = os.path.join(args.DATA_DIR_TGT, "blacklist.pkl")
    with open(blacklist_file, "rb") as f:
        blacklist = pkl.load(f)
    build_dataset(cie_list, syn_cie, blacklist)


def build_dataset(cie_list, syn_cie, blacklist):
    sets = ["_TRAIN", "_VALID", "_TEST"]
    if args.temporal:
        timestamp_counter = dict()
        timestamp_counter["start"] = Counter()
        timestamp_counter["end"] = Counter()
        timestamp_counter["duration"] = Counter()
        current_file = os.path.join(args.DATA_DIR_SRC, args.base_file)
        with ipdb.launch_ipdb_on_exception():
            with open(os.path.join(args.DATA_DIR_SRC, args.index_file), "rb") as f:
                index = pkl.load(f)
            for item in sets:
                with open(current_file + item + ".json", 'r') as f:
                    num_lines = sum(1 for line in f)
                with open(current_file + item + ".json", 'r') as f:
                    pbar = tqdm(f, total=num_lines)
                    dataset = []
                    for line in pbar:
                        current_person = json.loads(line)
                        new_p, new_tstp = build_new_person_tstp(current_person, index, timestamp_counter, selected_companies)
                        dataset.append(new_p)
                        timestamp_counter = new_tstp
                        pbar.update(1)
                ppl_file = os.path.join(args.DATA_DIR_TGT, 'ppl_mc_cie' + item + ".pkl")
                with open(ppl_file, 'wb') as fp:
                    pkl.dump(dataset, fp)
            tstp_file = os.path.join(args.DATA_DIR_TGT, "tstp_mc_cie_counter.pkl")
            with open(tstp_file, "wb") as tgt_file:
                pkl.dump(new_tstp, tgt_file)
    else:
        current_file = os.path.join(args.DATA_DIR_SRC, args.base_file)
        with ipdb.launch_ipdb_on_exception():
            with open(current_file, 'r') as f:
                num_lines = sum(1 for line in f)
            with open(current_file, 'r') as f:
                pbar = tqdm(f, total=num_lines)
                dataset = []
                for line in pbar:
                    current_person = json.loads(line)
                    jobs = current_person[1]
                    skills = current_person[2]
                    if len(jobs) >= args.MIN_JOB_COUNT and len(skills) > 0:
                        new_p = build_new_person(current_person, cie_list, syn_cie, blacklist)
                        if len(new_p[-1]) >= args.MIN_JOB_COUNT:
                            dataset.append(new_p)
                    pbar.update(1)

            ppl_file = os.path.join(args.DATA_DIR_TGT, "profiles_jobs_skills.pkl")
            with open(ppl_file, 'wb') as fp:
                pkl.dump(dataset, fp)


def build_new_person(person, cie_list, syn_cie, blacklist):
    person_id = person[0]
    industry = person[-1]
    skills = person[-3]
    ipdb.set_trace()
    education = person
    new_p = [person_id, skills, industry]
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
                    try:
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
                                                           job["description"], cie_list,  syn_cie)}
                            jobs.append(j)
                    except:
                        continue
    new_p.append(jobs)
    return new_p


# def build_new_person_tstp(person, index, tstp, mc_companies):
#     new_p = [person[0], person[-3], person[-1]]
#     jobs = []
#     with ipdb.launch_ipdb_on_exception():
#         for job in person[1]:
#             if 'company' in job.keys():
#                 threshold = min(len(job["company"].split(" ")), 5)
#                 tmp = job["company"].split(" ")[:threshold]
#                 normalized_name = [unidecode.unidecode(name.lower()) for name in tmp]
#                 company_name = " ".join(normalized_name)
#                 if company_name in mc_companies:
#                     end = handle_date(job)
#                     tstmp = pd.Timestamp.fromtimestamp(job["from_ts"])
#                     start = round(datetime.timestamp(tstmp.round("D").to_pydatetime()))
#                     if (end > 0) and (start > 0): # corresponds to the timestamp of 01/01/1970
#                         tstp["end"][end] += 1
#                         tstp["start"][start] += 1
#                         tstp["duration"][getDuration(job['from_ts'], end, "days")] += 1
#                         j = {'from': start,
#                              'to': end,
#                              'company': company_name,
#                              'job': word_seq_into_list_w_index(job["position"].split(' '), job["description"].split(' '), index)}
#                         jobs.append(j)
#         new_p.append(jobs)
#         return new_p, tstp


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
