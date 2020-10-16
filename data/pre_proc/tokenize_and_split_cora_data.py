import argparse
import os
import pickle as pkl
import random

import ipdb
import yaml
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from datetime import datetime


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(CFG["datadir"], "classifed_papers.pkl"), 'rb') as f:
            papers = pkl.load(f)
        papers_transformed = dict()
        for k in tqdm(papers.keys()):
            papers_transformed[k] = {}
            for handle in ["title", 'author', 'year', "editor", "note", "URL", "Keyword", "Affiliation", "Abstract"]:
                if handle in papers[k].keys():
                    papers_transformed[k][handle] = word_tokenize(papers[k][handle])
            for handle in ["filename", "class", "Reference"]:
                papers_transformed[k][handle] = papers[k][handle]
        randomized_indices = random.shuffle([i for i in papers_transformed.keys()])

        train, valid, test = []
        for k in randomized_indices:
            tmp = random.random()
            if tmp < .2:
                if tmp < .1:
                    test.append((k, papers_transformed[k]))
                else:
                    valid.append((k, papers_transformed[k]))
            else:
                train.append((k, papers_transformed[k]))

        print("Train ratio: " + str(100 * len(train) / len(papers_transformed)) + "%")
        print("Valid ratio: " + str(100 * len(valid) / len(papers_transformed)) + "%")
        print("Test ratio: " + str(100 * len(test) / len(papers_transformed)) + "%")

        input_file = "cora"

        with open(os.path.join(CFG["datadir"], input_file + "_TRAIN.pkl"), 'wb') as f:
            pkl.dump(train, f)
        with open(os.path.join(CFG["datadir"], input_file + "_VALID.pkl"), 'wb') as f:
            pkl.dump(valid, f)
        with open(os.path.join(CFG["datadir"], input_file + "_TEST.pkl"), 'wb') as f:
            pkl.dump(test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
