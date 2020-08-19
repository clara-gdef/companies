import os
import argparse
import ipdb
import yaml
import pickle as pkl
from sklearn.metrics import plot_confusion_matrix


def main(hparams):
    data = {}
    for data_agg_type in hparams.data_agg_type:
        file_path = os.path.join(CFG["datadir"], "OUTPUTS_" +
                                 hparams.model_type + "_" +
                                 hparams.rep_type + "_" +
                                 data_agg_type + "_" +
                                 hparams.input_type +
                                 "_bs" + str(hparams.b_size) + ".pkl")
        with open(file_path, 'rb') as f:
            data[data_agg_type] = pkl.load(f)
    ipdb.set_trace()


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='sk')
    parser.add_argument("--b_size", type=int, default=64)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=False)
    parser.add_argument("--data_agg_type", type=str, default=["avg", "max", "sum"])
    parser.add_argument("--model_type", type=str, default='disc_poly')
    hparams = parser.parse_args()
    main(hparams)
