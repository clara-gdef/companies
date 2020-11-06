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
from data.datasets import DiscriminativeCoraDataset
from models.classes import InstanceClassifierDiscCora
from models.classes.InstanceClassifierDisc import get_metrics_at_k, get_metrics
from utils.models import get_model_params, collate_for_disc_spe_model_cora
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        main(hparams)


def main(hparams):
    xp_title = "b4Traing_" + hparams.model_type + "_" + hparams.ft_type + "_" + hparams.input_type

    datasets = load_datasets(hparams, CFG, ["TEST"])
    dataset_test = datasets[0]
    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_disc_spe_model_cora, num_workers=8,
                             shuffle=True)

    in_size, out_size = get_model_params(hparams, 300, len(dataset_test.track_rep))

    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "num_tracks": len(dataset_test.track_rep),
                 "input_type": hparams.input_type,
                 "ft_type": hparams.ft_type}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDiscCora(**arguments)
    print("Model Loaded.")
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
                    res_k = get_metrics_at_k(predicted_classes[:, :k], labels[handle], num_c, handle + "_@" + str(k),
                                             offset)
                    res = {**res, **res_k}
            gen_res = get_general_results_poly(preds, labels, len(dataset_test.track_rep))
        else:
            handle = "cie"
            offset = 0
            num_c = 207
            predicted_classes = torch.argsort(preds, dim=-1, descending=True)
            for k in [1, 10]:
                res_k = get_metrics_at_k(predicted_classes[:, :k], labels, num_c, handle + "_@" + str(k), offset)
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


def load_datasets(hparams, CFG, splits):
    datasets = []
    common_hparams = {
        "datadir": CFG["gpudatadir"],
        "track_file": CFG["rep"]["cora"]["tracks"],
        "paper_file": CFG["rep"]["cora"]["papers"]["emb"],
        "ft_type": hparams.ft_type,
        "subsample": 0,
        "load": hparams.load_dataset == "True"
    }
    for split in splits:
        datasets.append(DiscriminativeCoraDataset(**common_hparams, split=split))

    return datasets


def get_well_classified_outputs(res_dict):
    well_classified = {}
    for k, offset in zip(res_dict["preds"].keys(), [0, 207, 237]):
        predicted_classes = get_predicted_classes(res_dict["preds"][k], offset)
        print("F1 for " + str(k) + ': ' + str(
            f1_score(predicted_classes, res_dict["labels"][k], average="weighted", zero_division=0) * 100))
        print("Acc for " + str(k) + ': ' + str(accuracy_score(predicted_classes, res_dict["labels"][k]) * 100))
        print("Prec for " + str(k) + ': ' + str(
            precision_score(predicted_classes, res_dict["labels"][k], average="weighted", zero_division=0) * 100))
        print("Rec for " + str(k) + ': ' + str(
            recall_score(predicted_classes, res_dict["labels"][k], average="weighted", zero_division=0) * 100))

        good_outputs, labels, good_idx = find_well_classified_outputs(predicted_classes, res_dict["preds"][k],
                                                                      res_dict["labels"][k], res_dict["indices"])
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
    for preds, labs in tqdm(zip(itertools.chain(cie_preds_at_k, clus_preds_at_k, dpt_preds_at_k), chained_labels),
                            desc="Computing at k=10..."):
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
    parser.add_argument("--ft_type", type=str, default='pt')
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--model_type", type=str, default="cora_disc_spe_std")
    parser.add_argument("--load_dataset", type=str, default="False")
    hparams = parser.parse_args()
    init(hparams)
