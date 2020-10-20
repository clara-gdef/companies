import argparse
import os
import pickle as pkl
import yaml
import ipdb
from tqdm import tqdm
import fastText


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with open(os.path.join(CFG["datadir"], args.input_file + "_TRAIN.pkl"), 'rb') as f:
        data_train = pkl.load(f)

    with ipdb.launch_ipdb_on_exception():
        tgt_file = make_txt_file(data_train)

        ft_model = fastText.train_unsupervised(tgt_file, dim=300)

        ft_model.save_model(os.path.join(CFG["modeldir"], args.model_file))


def make_txt_file(data_train):
    tgt_file = os.path.join(CFG["datadir"], "ft_unsupervised_input.txt")
    with open(tgt_file, "a") as f:
        ipdb.set_trace()
        for data in tqdm(data_train, "parsing train data..."):
            for j in data[-1]:
                f.write(" ".join(j["job"]))
                f.write("\n")
            if len(data[-2]) > 0:
                for e in data[-2][0]:
                    f.write(" ".join(e))
                    f.write("\n")
    return tgt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="cora")
    parser.add_argument("--model_file", type=str, default="ft_fs_cora.bin")
    args = parser.parse_args()
    main(args)
