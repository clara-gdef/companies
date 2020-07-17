import torch
import os
import ipdb
import pickle as pkl
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from data.datasets import DiscriminativePolyvalentDataset

from models.classes import InstanceClassifier


def main(args):
    with ipdb.launch_ipdb_on_exception():
        dataset = load_dataset(args)
        in_size, out_size = get_model_params(dataset.user_dim, dataset.bag_dim)
        train_loader = DataLoader(dataset, batch_size=args.b_size)
        model = InstanceClassifier(in_size, out_size)
        trainer = pl.Trainer(gpus=args.gpus, precision=16)
        trainer.fit(model, train_loader)


def load_dataset(args):
    dataset = DiscriminativePolyvalentDataset(CFG["gpudatadir"],
                                              ppl_file=CFG["rep"][args.rep_type]["total"] + "_TRAIN.pkl",
                                              rep_type=args.rep_type,
                                              cie_reps_file=CFG["rep"]["cie"] + args.data_agg_type + ".pkl",
                                              clus_reps_file=CFG["rep"]["clus"] + args.data_agg_type + ".pkl",
                                              dpt_reps_file=CFG["rep"]["dpt"] + args.data_agg_type + ".pkl",
                                              agg_type=args.data_agg_type,
                                              load=True)
    return dataset


def get_model_params(rep_dim, bag_dim):
    if args.input_type == "hadamard":
        in_size = None
        out_size = None
    elif args.input_type == "matMul":
        in_size = None
        out_size = None
    elif args.input_type == "matMulExt":
        in_size = None
        out_size = None
    elif args.input_type == "concat":
        in_size = None
        out_size = None

    return in_size, out_size


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='sk')
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--b_size", type=int, default=64)
    parser.add_argument("--input_type", type=str, default="hadamard")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    args = parser.parse_args()
    main(args)
