import ipdb
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from data.datasets import DiscriminativePolyvalentDataset
from models.classes import InstanceClassifier
from utils.models import collate_for_disc_poly_model


def main(args):
    with ipdb.launch_ipdb_on_exception():
        dataset_train, dataset_valid, dataset_test = load_datasets(args)
        in_size, out_size = get_model_params(len(dataset_train), len(dataset_train.bag_rep))

        train_loader = DataLoader(dataset_train, batch_size=args.b_size, collate_fn=collate_for_disc_poly_model)
        valid_loader = DataLoader(dataset_valid, batch_size=args.b_size, collate_fn=collate_for_disc_poly_model)
        valid_loader = DataLoader(dataset_test, batch_size=args.b_size, collate_fn=collate_for_disc_poly_model)

        print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
        model = InstanceClassifier(in_size, out_size, args.input_type)
        print("Model Loaded.")
        trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs)
        trainer.fit(model, train_loader, args.lr)


def load_datasets(args):
    datasets = []
    common_args = {
        "data_dir": CFG["gpudatadir"],
        "ppl_file": CFG["rep"][args.rep_type]["total"],
        "rep_type": args.rep_type,
        "cie_reps_file": CFG["rep"]["cie"] + args.data_agg_type + ".pkl",
        "clus_reps_file": CFG["rep"]["clus"] + args.data_agg_type + ".pkl",
        "dpt_reps_file": CFG["rep"]["dpt"] + args.data_agg_type + ".pkl",
        "agg_type": args.data_agg_type,
        "load": False
    }
    for split in ["TRAIN", "VALID", "TEST"]:
        datasets.append(DiscriminativePolyvalentDataset(**common_args, split=split))

    return datasets[0], datasets[1], datasets[2]


def get_model_params(rep_dim, num_bag):
    out_size = num_bag
    if args.input_type == "hadamard" or args.input_type == "concat":
        in_size = rep_dim * num_bag
    elif args.input_type == "matMul":
        in_size = num_bag
    else:
        raise Exception("Wrong input data specified: " + str(args.input_type))

    return in_size, out_size


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='sk')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=64)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
