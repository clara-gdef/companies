import os
import argparse
import pickle as pkl

import yaml
from tqdm import tqdm
import ipdb

def main(args):
    
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
        classes = pkl.load(f)
    with ipdb.launch_ipdb_on_exception():
        sets = ["_TRAIN", "_VALID", "_TEST"]
        for item in sets:
            tracks_dict = {}
            ppl_file = os.path.join(CFG["gpudatadir"], "cora" + item + ".pkl")
            with open(ppl_file, 'rb') as fp:
                data = pkl.load(fp)
            ipdb.set_trace()
            for person in tqdm(data, desc="Build track for " + str(item)):
                build_track_info_from_person(person, tracks_dict, classes)
            tgt_file = os.path.join(CFG["gpudatadir"], 'track' + item + ".pkl")
            with open(tgt_file, 'wb') as fc:
                pkl.dump(tracks_dict, fc)


def build_track_info_from_person(person, tracks_dict, classes):
    identifier, skills, industry, jobs = person
    if len(jobs) > 0:
        for job in jobs:
            tracks_dict[track] = {}

            for year in range(start, end):
                if track in track_list:
                    if year not in tracks_dict[track].keys():
                        tracks_dict[track][year] = []
                    tracks_dict[track][year].append([identifier, job["job"], skills, industry])
    return tracks_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
