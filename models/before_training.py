import ipdb
import torch
import os
import argparse
from torch.utils.data import DataLoader
import yaml
import pickle as pkl
from data.datasets import DiscriminativePolyvalentDataset
from models.classes import InstanceClassifierDisc
from utils.models import collate_for_disc_poly_model, get_model_params


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        train(hparams)


def train(hparams):
    xp_title = "disc_poly_b4_training_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_" + \
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
                 "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDisc(**arguments)
    print("Model Loaded.")
    print("Starting eval for " + xp_title + "...")
    preds_and_labels = model.get_outputs_and_labels(test_loader)
    well_classified = get_well_classified_outputs(preds_and_labels)
    with open(os.path.join(CFG["gpudatadir"], "OUTPUTS_well_classified_" + xp_title), 'wb') as f:
        pkl.dump(preds_and_labels, f)


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


def get_well_classified_outputs(res_dict):
    well_classified = {}
    for k in res_dict["preds"].keys():
        preds = get_predicted_classes(res_dict["preds"][k])
        good_outputs, labels, good_idx = find_well_classified_outputs(preds, res_dict["labels"][k], res_dict["indices"])
        well_classified[k] = {v: k for v, k in zip(good_idx, labels)}
    ipdb.set_trace()


def get_predicted_classes(outvectors):
    return [i.item() for i in torch.argmax(outvectors, dim=-1)]


def find_well_classified_outputs(preds, labels, idx):
    indices = []
    for index, pred, lab in enumerate(zip(preds, labels)):
        if pred == lab:
            indices.append(index)
    return preds[indices], labels[indices], idx[indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=1)
    parser.add_argument("--middle_size", type=int, default=20)
    parser.add_argument("--input_type", type=str, default="b4Training")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=20)
    hparams = parser.parse_args()
    main(hparams)
