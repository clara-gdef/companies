import yaml
import os
import pickle as pkl
import ipdb
from tqdm import tqdm


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
        classes = pkl.load(f)
    ipdb.set_trace()
    

if __name__ == "__main__":
    main()
