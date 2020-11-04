import os
import argparse
import pickle as pkl
import fastText
import yaml
from tqdm import tqdm
import ipdb


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
        classes = pkl.load(f)

    print("Loading word vectors...")
    if args.ft_type == "fs":
        embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_cora.bin"))
    else:
        embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_en.bin"))
    print("Word vectors loaded.")

    for split in ["TRAIN", 'VALID', "TEST"]:
        paper_file = os.path.join(CFG["gpudatadir"], "cora_" + split + ".pkl")
        with open(paper_file, 'rb') as fp:
            data = pkl.load(fp)
        for paper in tqdm(data, desc="Parsing articles for split " + split + "..."):
            identifier = paper[0]
            profiles = paper[1]["Abstract"]
            ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default="ft_en.bin")
    args = parser.parse_args()
    main(args)
