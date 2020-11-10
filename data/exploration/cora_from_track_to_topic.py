import yaml
import os
import pickle as pkl
import ipdb
from collections import Counter
from tqdm import tqdm


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
        class_dict = pkl.load(f)

    rev_class_dict = {v: k for k, v in class_dict.items()}
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data_train = pkl.load(f)

    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "VALID.pkl")
    with open(paper_file, 'rb') as f:
        data_valid = pkl.load(f)

    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "TEST.pkl")
    with open(paper_file, 'rb') as f:
        data_test = pkl.load(f)

    class_count = Counter()
    for paper in tqdm(zip(data_train, data_valid, data_test)):
        high_lvl_class = rev_class_dict[paper[1]["class"]].split("/")[1]
        class_count[high_lvl_class] += 1

    ipdb.set_trace()


if __name__ == "__main__":
    main()
