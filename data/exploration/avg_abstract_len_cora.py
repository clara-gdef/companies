import os
import pickle as pkl
import ipdb
import yaml


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["emb"] + "_fs_TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data = pkl.load(f)
    num_sentences = []
    for paper in data:
        ipdb.set_trace()
        num_sentences.append(len(paper))


if __name__ == "__main__":
    main()
