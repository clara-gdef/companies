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
            tmp = pkl.load(fp)
        data.extend(tmp)
        paper_lookup_split = build_article_lookup(tmp)
        with open(os.path.join(CFG["gpudatadir"], "article_lookup" + item + ".pkl"), "wb") as f:
            pkl.dump(paper_lookup_split, f)

    paper_lookup_global = build_article_lookup(data)

    with open(os.path.join(CFG["gpudatadir"], "article_lookup_global.pkl"), "wb") as f:
        pkl.dump(paper_lookup_global, f)

    track_lookup = build_track_lookup(data, classes, paper_lookup_global)
    with open(os.path.join(CFG["gpudatadir"], "track_dict.pkl"), 'wb') as f:
        pkl.dump(track_lookup, f)


def build_article_lookup(articles):
    lookup = {}
    for article in tqdm(articles, desc="Building lookup for articles..."):
        lookup[article[0]] = article[1]
    return lookup


def build_track_lookup(articles, classes, paper_lookup_global):
    track_lookup = {}
    for article in tqdm(articles, desc="Building track lookup..."):
        track = article[1]["class"]
        if track not in track_lookup.keys():
            track_lookup[track] = {"id": [], "profiles": []}
        track_lookup[track]["id"].append(article[0])
        track_lookup[track]["profiles"].append(paper_lookup_global[article[0]])
    assert len(track_lookup) == len(classes)
    return track_lookup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
