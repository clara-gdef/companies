import os
import pickle as pkl
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
        ipdb.set_trace()

if __name__ == "__main__":
    main()
