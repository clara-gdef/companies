import os
import pickle as pkl
import ipdb
import yaml
from tqdm import tqdm
from datetime import datetime


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    ppl_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu")
    tgt_file = os.path.join(CFG["datadir"], "profiles_jobs_skills_edu_delta_t")
    with ipdb.launch_ipdb_on_exception():
        for split in ["_TRAIN", "_VALID", "_TEST"]:
            with open(ppl_file + split + ".pkl", 'rb') as fp:
                data = pkl.load(fp)
            dataset = build_dataset(data, split)
            with open(tgt_file + split + ".pkl", 'wb') as f:
                pkl.dump(dataset, f)


def build_dataset(data, split):
    all_durations_days = []
    faulty_profiles = set()
    good_profiles = []
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
            good_profiles.append(person)
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
    main()
