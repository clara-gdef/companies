import itertools
from tqdm import tqdm
import numpy as np
import ipdb
import torch
import os
import argparse
from torch.utils.data import DataLoader
import yaml
import pickle as pkl
from data.datasets import DiscriminativeSpecializedDataset, DiscriminativePolyvalentDataset
from models.classes import InstanceClassifierDisc
from models.classes.InstanceClassifierDisc import get_metrics_at_k, get_metrics
from utils.models import collate_for_disc_poly_model, get_model_params, collate_for_disc_spe_model
from sklearn.metrics import f1_score,  accuracy_score, precision_score, recall_score


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        main(hparams)


def main(hparams):
    if hparams.type == "poly":
        num_classes = 207 + 30 + 5888
    else:
        num_classes = 207

    datasets = load_datasets(hparams, ["TEST"], hparams.load_dataset)
    dataset_train, dataset_test = datasets[0], datasets[1]
    in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, len(dataset_train.bag_rep))

    if hparams.type == "poly":
        xp_title = "disc_poly_b4_training_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_" + \
                   str(hparams.b_size) + "_" + str(hparams.lr)
        test_loader = DataLoader(dataset_test, batch_size=hparams.b_size, collate_fn=collate_for_disc_poly_model,
                                 num_workers=8)
        arguments = {'in_size': in_size,
                     'out_size': out_size,
                     'hparams': hparams,
                     'dataset': dataset_train,
                     "bag_type": None,
                     'datadir': CFG["gpudatadir"],
                     'desc': xp_title,
                     "wd": hparams.wd,
                     "middle_size": hparams.middle_size}
    else:
        xp_title = "disc_spe_b4_training_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_" + \
                   str(hparams.b_size) + "_" + str(hparams.lr)
        test_loader = DataLoader(dataset_test, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                                 num_workers=8)
        arguments = {'in_size': in_size,
                     'out_size': out_size,
                     'hparams': hparams,
                     'dataset': dataset_train,
                     "bag_type": "cie",
                     'datadir': CFG["gpudatadir"],
                     'desc': xp_title,
                     "wd": hparams.wd,
                     "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDisc(**arguments)
    print("Model Loaded.")
    print("Starting eval for " + xp_title + "...")
    preds_and_labels = model.get_outputs_and_labels(test_loader)
    if hparams.eval_top_k:
        res = {}
        preds = preds_and_labels["preds"]
        labels = preds_and_labels["labels"]

        if hparams.type == "poly":
            for handle, offset, num_c in zip(["cie", "clus", "dpt"], [0, 207, 237], [207, 30, 5888]):
                predicted_classes = torch.argsort(preds[handle], dim=-1, descending=True)
                for k in [1, 10]:
                    res_k = get_metrics_at_k(predicted_classes[:, :k], labels[handle], num_c, handle + "_@"+str(k), offset)
                    res = {**res, **res_k}
            gen_res = get_general_results_poly(preds, labels, num_classes)
        else:
            handle = "cie"
            offset = 0
            num_c = 207
            predicted_classes = torch.argsort(preds, dim=-1, descending=True)
            for k in [1, 10]:
                res_k = get_metrics_at_k(predicted_classes[:, :k], labels, num_c, handle + "_@"+str(k), offset)
                res = {**res, **res_k}
            gen_res = res

        res = {**res, **gen_res}

        print(sorted(res.items()))

        with open(os.path.join(CFG["gpudatadir"], "OUTPUTS_well_classified_topK_" + xp_title), 'wb') as f:
            pkl.dump(res, f)
    else:
        well_classified = get_well_classified_outputs(preds_and_labels)
        with open(os.path.join(CFG["gpudatadir"], "OUTPUTS_well_classified_" + xp_title), 'wb') as f:
            pkl.dump(well_classified, f)


def load_datasets(hparams, splits, load):
    datasets = []
    if hparams.type == "poly":
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "ppl_file": CFG["rep"][hparams.rep_type]["total"],
            "rep_type": hparams.rep_type,
            "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type,
            "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type,
            "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type,
            "agg_type": hparams.data_agg_type,
            "load": load,
            "subsample": 0,
            "standardized": False
        }
        if hparams.standardized == "True":
            print("Loading standardized datasets...")
            common_hparams["standardized"] = True

        for split in splits:
            datasets.append(DiscriminativePolyvalentDataset(**common_hparams, split=split))
    else:
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "rep_type": hparams.rep_type,
            "bag_type": "cie",
            "agg_type": hparams.data_agg_type,
            "subsample": 0,
            "standardized": False
        }
        if hparams.standardized == "True":
            print("Loading standardized datasets...")
            common_hparams["standardized"] = True

        for split in splits:
            datasets.append(DiscriminativeSpecializedDataset(**common_hparams, split=split))
    return datasets


def get_well_classified_outputs(res_dict):
    well_classified = {}
    for k, offset in zip(res_dict["preds"].keys(), [0, 207, 237]):
        predicted_classes = get_predicted_classes(res_dict["preds"][k], offset)
        print("F1 for " + str(k) + ': ' + str(f1_score(predicted_classes, res_dict["labels"][k], average="weighted", zero_division=0) * 100))
        print("Acc for " + str(k) + ': ' + str(accuracy_score(predicted_classes, res_dict["labels"][k]) * 100))
        print("Prec for " + str(k) + ': ' + str(precision_score(predicted_classes, res_dict["labels"][k], average="weighted", zero_division=0) * 100))
        print("Rec for " + str(k) + ': ' + str(recall_score(predicted_classes, res_dict["labels"][k], average="weighted", zero_division=0) * 100))

        good_outputs, labels, good_idx = find_well_classified_outputs(predicted_classes, res_dict["preds"][k], res_dict["labels"][k], res_dict["indices"])
        well_classified[k] = {v: (k.numpy(), j.item()) for v, k, j in zip(good_idx, good_outputs, labels)}
    return well_classified


def get_predicted_classes(outvectors, offset):
    return [i.item() + offset for i in torch.argmax(outvectors, dim=-1)]


def find_well_classified_outputs(predicted_classes, preds, labels, idx):
    indices = []
    for index, tup in enumerate(zip(predicted_classes, labels)):
        if tup[0] == tup[1].item():
            indices.append(index)
    good_preds = [preds[i] for i in indices]
    good_labels = [labels[i] for i in indices]
    good_indices = [idx[i] for i in indices]
    return good_preds, good_labels, good_indices


def get_general_results_poly(preds, labels, num_classes):
    all_labels = []
    for tup in zip(labels["cie"], labels["clus"], labels["dpt"]):
        all_labels.append([tup[0].item(), tup[1].item(), tup[2].item()])
    cie_preds_max = [i.item() for i in torch.argmax(preds["cie"], dim=1)]
    clus_preds_max = [i.item() + 207 for i in torch.argmax(preds["clus"], dim=1)]
    dpt_preds_max = [i.item() + 237 for i in torch.argmax(preds["dpt"], dim=1)]
    all_preds_max = []
    for tup in zip(cie_preds_max, clus_preds_max, dpt_preds_max):
        all_preds_max.append([tup[0], tup[1], tup[2]])

    res_all = get_metrics(np.array(all_preds_max).reshape(-1, 1), np.array(all_labels).reshape(-1, 1),
                                num_classes, "all", 0)

    cie_preds_at_k = [i for i in torch.argsort(preds["cie"], dim=-1, descending=True)]
    clus_preds_at_k = [i + 207 for i in torch.argsort(preds["clus"], dim=-1, descending=True)]
    dpt_preds_at_k = [i + 237 for i in torch.argsort(preds["dpt"], dim=-1, descending=True)]

    all_preds_k = []
    chained_labels = [i.item() for i in itertools.chain(labels["cie"], labels["clus"], labels["dpt"])]
    for preds, labs in tqdm(zip(itertools.chain(cie_preds_at_k, clus_preds_at_k, dpt_preds_at_k), chained_labels), desc="Computing at k=10..."):
        if labs in preds[:10]:
            all_preds_k.append(labs)
        else:
            if type(preds) == torch.Tensor:
                all_preds_k.append(preds[0].item())
            else:
                all_preds_k.append(preds)

    res_all_k = get_metrics(all_preds_k, chained_labels, num_classes, "all_@10", 0)
    return {**res_all_k, **res_all}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=1)
    parser.add_argument("--middle_size", type=int, default=20)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--input_type", type=str, default="b4Training")
    parser.add_argument("--type", type=str, default="spe")
    parser.add_argument("--load_dataset", type=bool, default=False)
    parser.add_argument("--eval_top_k", type=bool, default=True)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--standardized", type=str, default="True")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=20)
    hparams = parser.parse_args()
    init(hparams)
