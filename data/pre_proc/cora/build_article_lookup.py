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
    sets = ["_TRAIN", "_VALID", "_TEST"]
    data = []
    for item in sets:
        ppl_file = os.path.join(CFG["gpudatadir"], "cora" + item + ".pkl")
        with open(ppl_file, 'rb') as fp:
            data.extend(pkl.load(fp))
    track_lookup = build_ppl_lookup(data, classes)
    with open(os.path.join(CFG["gpudatadir"], "track_dict.pkl"), 'wb') as f:
        pkl.dump(track_lookup, f)


def build_track_lookup(articles, classes):
    track_lookup = {}
    for article in tqdm(articles, desc="Building track lookup..."):
        track = article[1]["class"]
        if track not in track_lookup.keys():
            track_lookup[track] = []
        track_lookup[track].append(article[0])
    assert len(track_lookup) == len(classes)
    return track_lookup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
