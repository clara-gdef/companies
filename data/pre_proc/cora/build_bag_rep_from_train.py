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

    ppl_file = os.path.join(CFG["gpudatadir"], "cora_TRAIN.pkl")
    with open(ppl_file, 'rb') as fp:
        data = pkl.load(fp)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default="ft_en.bin")
    args = parser.parse_args()
    main(args)
