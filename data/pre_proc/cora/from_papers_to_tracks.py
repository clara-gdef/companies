import os
import argparse
import pickle as pkl
from tqdm import tqdm
from utils.Utils import date_to_year
import ipdb

def main(args):
    with ipdb.launch_ipdb_on_exception():
        sets = ["_TRAIN", "_VALID", "_TEST"]
        for item in sets:
            tracks_dict = {}
            ppl_file = os.path.join(args.DATA_DIR, args.input_file + item + ".pkl")
            with open(ppl_file, 'rb') as fp:
                data = pkl.load(fp)
            ipdb.set_trace()
            for person in tqdm(data, desc="Build track for " + str(item)):
                build_track_info_from_person(person, tracks_dict)
            tgt_file = os.path.join(args.DATA_DIR, 'track' + item + ".pkl")
            with open(tgt_file, 'wb') as fc:
                pkl.dump(tracks_dict, fc)


def build_track_info_from_person(person, tracks_dict):
    identifier, skills, industry, jobs = person
    if len(jobs) > 0:
        for job in jobs:
            tracks_dict[track] = {}
            start = date_to_year(job["from"])
            end = date_to_year(job["to"])
            for year in range(start, end):
                if track in track_list:
                    if year not in tracks_dict[track].keys():
                        tracks_dict[track][year] = []
                    tracks_dict[track][year].append([identifier, job["job"], skills, industry])
    return tracks_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default="/local/gainondefor/work/lip6/data/tracks")
    parser.add_argument("--input_file", type=str, default="cora")
    args = parser.parse_args()
    main(args)
