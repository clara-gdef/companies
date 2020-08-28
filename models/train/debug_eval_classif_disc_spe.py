import os
import torch

import ipdb
import argparse
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from data.datasets import DiscriminativeSpecializedDataset
from models.classes import DebugClassifierDisc
from utils.models import collate_for_disc_spe_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


num_cie = 207
num_clus = 30
num_dpt = 5888


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        # Load datasets
        xp_title = "disc_spe_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" +\
                   hparams.input_type + "_bs1"
        datasets = load_datasets(hparams, ["TEST"])
        dataset_test = datasets[0]

        # initiate model
        in_size, out_size = dataset_test.get_num_bag(), dataset_test.get_num_bag()
        test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_disc_spe_model,
                                  num_workers=16, shuffle=True)
        arguments = {'in_size': in_size,
                     'out_size': out_size,
                     'hparams': hparams,
                     'dataset': dataset_test,
                     'datadir': CFG["gpudatadir"],
                     'desc': xp_title}
        print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
        model = DebugClassifierDisc(**arguments)

        # set up file writers
        file_name = "debug_disc_spe_" + str(hparams.rep_type) + "_" + str(hparams.bag_type) + "_" + str(hparams.lr)
        file_path = os.path.join(CFG["modeldir"], file_name)

        model.load_state_dict(torch.load(file_path))
        test(hparams, model.cuda(), test_loader)


def test(hparams, model, test_loader):
    b4_training = []
    outputs = []
    labels = []
    ids = []
    preds = []
    for identifier, ppl, tmp_labels, bag_rep in tqdm(test_loader, desc="Testing..."):
        bag_rep = torch.transpose(bag_rep, 1, 0)
        input_tensor = torch.matmul(ppl, bag_rep).cuda()
        output = model(input_tensor)

        ids.append(identifier)
        b4_training.append(torch.argmax(input_tensor, dim=1).item())
        outputs.append(output)
        preds.append(torch.argmax(output, dim=1).item())
        labels.append(tmp_labels[0])
    preds, cm, res_trained, res_b4_training = test_for_bag(preds, labels, b4_training, 0, get_num_classes(hparams.bag_type))
    prec, rec = get_average_metrics(res)
    ipdb.set_trace()
    return {"acc": res["accuracy"],
            "precision": prec,
            "recall": rec}


def get_num_classes(bag_type):
    if type == "spe":
        if bag_type == "cie":
            return num_cie
        elif bag_type == "clus":
            return num_clus
        elif bag_type == "dpt":
            return num_dpt
    else:
        return num_cie + num_clus + num_dpt


def test_for_bag(preds, labels, b4_training, offset, num_classes):
    predicted_classes = [i + offset for i in preds]
    preds_b4 = [i + offset for i in b4_training]
    cm = confusion_matrix(labels.cpu().numpy(), np.asarray(predicted_classes))
    results_trained = classification_report(predicted_classes, labels, output_dict=True, labels=range(num_classes))
    res_b4_training = classification_report(preds_b4, labels, output_dict=True, labels=range(num_classes))
    return predicted_classes, cm, results_trained, res_b4_training


def get_average_metrics(res_dict):
    precision = []
    recall = []
    numerical_keys = [i for i in res_dict.keys()][:-3]
    for k in numerical_keys:
        precision.append(res_dict[k]["precision"])
        recall.append(res_dict[k]["recall"])
    return np.mean(precision), np.mean(recall)


def load_datasets(hparams, splits):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "rep_type": hparams.rep_type,
        "agg_type": hparams.data_agg_type,
        "bag_type": hparams.bag_type,
    }
    for split in splits:
        datasets.append(DiscriminativeSpecializedDataset(**common_hparams, split=split))

    return datasets



if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--lr", type=float, default=1e-7)
    hparams = parser.parse_args()
    main(hparams)

