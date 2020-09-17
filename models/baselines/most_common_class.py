import ipdb
import torch
import os
import argparse
from torch.utils.data import DataLoader
import yaml
import pickle as pkl
from data.datasets import DiscriminativePolyvalentDataset
from models.classes import InstanceClassifierDisc
from models.classes.InstanceClassifierDisc import get_metrics_at_k
from utils.models import collate_for_disc_poly_model, get_model_params
from sklearn.metrics import f1_score,  accuracy_score, precision_score, recall_score


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        main(hparams)


def main(hparams):
    xp_title = "disc_poly_most_common_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_" + \
               str(hparams.b_size) + "_" + str(hparams.lr)
    datasets = load_datasets(hparams, ["TRAIN", "TEST"], hparams.load_dataset)
    dataset_train, dataset_test = datasets[0], datasets[1]
    in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, len(dataset_train.bag_rep))
    test_loader = DataLoader(dataset_test, batch_size=hparams.b_size, collate_fn=collate_for_disc_poly_model,
                             num_workers=8)
    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'dataset': dataset_train,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "wd": hparams.wd,
                 "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDisc(**arguments)
    print("Model Loaded.")
    print("Starting eval for " + xp_title + "...")
    most_common_classes = {}
    tgt_file = os.path.join(CFG["gpudatadir"], "most_common_classes_")
    for handle in ["cie", 'clus', "dpt"]:
        with open(tgt_file + handle + ".pkl", "rb") as f:
            tmp = pkl.load(f)
        most_common_classes[handle] = [i[0] for i in tmp]

    preds_and_labels = model.get_outputs_and_labels(test_loader)
    labels = preds_and_labels["labels"]

    res = {}
    for handle, num_c in zip(["cie", "clus", "dpt"], [207, 30, 5888]):
        predicted_classes = torch.LongTensor(most_common_classes[handle]).expand(len(dataset_test), -1)
        for k in [1, 10]:
            res_k = get_metrics_at_k(predicted_classes[:, :k], labels[handle], num_c, handle + "_@"+str(k), 0)
            res = {**res, **res_k}
    print(sorted(res.items()))


def load_datasets(hparams, splits, load):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "ppl_file": CFG["rep"][hparams.rep_type]["total"],
        "rep_type": hparams.rep_type,
        "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type + ".pkl",
        "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type + ".pkl",
        "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type + ".pkl",
        "agg_type": hparams.data_agg_type,
        "load": load
    }
    for split in splits:
        datasets.append(DiscriminativePolyvalentDataset(**common_hparams, split=split))

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=1)
    parser.add_argument("--middle_size", type=int, default=20)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--input_type", type=str, default="b4Training")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--eval_top_k", type=bool, default=True)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=20)
    hparams = parser.parse_args()
    init(hparams)
