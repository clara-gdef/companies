import yaml
import os
import pickle as pkl
import ipdb
from collections import Counter
from tqdm import tqdm
from itertools import chain

def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with ipdb.launch_ipdb_on_exception():
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

        hl_class_count = Counter()
        class_count = Counter()
        for paper in tqdm(chain(data_train, data_valid, data_test)):
            class_count[paper[1]["class"]] += 1
            high_lvl_class = paper[1]["class"].split("/")[1]
            hl_class_count[high_lvl_class] += 1

        high_lvl_classes = sorted(hl_class_count.keys())
        hl_class_dict = {name: num for num, name in enumerate(high_lvl_classes)}

        with open(os.path.join(CFG["gpudatadir"], "cora_high_level_classes_dict.pkl"), 'wb') as f:
            pkl.dump(hl_class_dict, f)

        track_to_hl_dict = {}
        for low_level_class in class_count.keys():
            ll_class_ind = rev_class_dict[low_level_class]
            high_lvl_class = low_level_class.split("/")[1]
            track_to_hl_dict[ll_class_ind] = hl_class_dict[high_lvl_class]
        with open(os.path.join(CFG["gpudatadir"], "cora_track_to_hl_classes_map.pkl"), 'wb') as f:
            pkl.dump(track_to_hl_dict, f)

    ipdb.set_trace()


if __name__ == "__main__":
    main()
