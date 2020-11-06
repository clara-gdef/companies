import os
import pickle as pkl
import ipdb
import numpy as np
import yaml
from tqdm import tqdm


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["emb"] + "_fs_TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data = pkl.load(f)
    num_sentences = []
    for paper in tqdm(data):
        num_sentences.append(len(paper["sentences_emb"]))
    print(np.percentile(num_sentences, 90))

if __name__ == "__main__":
    main()
