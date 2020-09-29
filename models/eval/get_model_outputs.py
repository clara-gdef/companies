import os

import ipdb
import argparse

from torch.utils.data import DataLoader
import yaml
import torch
import pickle as pkl
import glob
from data.datasets import DiscriminativeSpecializedDataset, DiscriminativePolyvalentDataset
from models.classes import InstanceClassifierDisc
from utils.models import collate_for_disc_spe_model, collate_for_disc_poly_model, get_model_params


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        xp_title = hparams.model_type + "_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size)

        if hparams.model_type.split("_")[-1] == "spe":
            collate_fn = collate_for_disc_spe_model
        else:
            collate_fn = collate_for_disc_poly_model
        datasets = load_datasets(hparams, CFG, ["TRAIN"])
        dataset_train = datasets[0]
        in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, len(dataset_train.bag_rep))

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

        model.eval()

        dataset = load_datasets(hparams, CFG, ["TEST"])
        test_loader = DataLoader(dataset[0], batch_size=1, collate_fn=collate_for_disc_spe_model, num_workers=0)
        model_name = hparams.model_type + "/" + hparams.bag_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + \
                     "/" + hparams.input_type + "/" + str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(
            hparams.wd)
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)

        if hparams.input_type != "b4Training":
            model_path = os.path.join(CFG['modeldir'], model_name)
            model_files = glob.glob(os.path.join(model_path, "*"))
            latest_file = max(model_files, key=os.path.getctime)
            print("Evaluating model: " + str(latest_file))
            model.load_state_dict(torch.load(latest_file)["state_dict"])

        test_res = model.get_outputs_and_labels(test_loader)
        tgt_file = os.path.join(CFG["gpudatadir"], "OUTPUTS_" + xp_title)
        with open(tgt_file + "_TEST.pkl", 'wb') as f:
            pkl.dump(test_res, f)

        if hparams.well_classified == "True":
            well_classif_indices = []
            for ind, (pred, lab) in enumerate(zip(test_res["preds"], test_res["labels"])):
                if torch.argmax(pred, dim=-1).item() == lab.item():
                    well_classif_indices.append(test_res["indices"][ind])
            tgt_file = os.path.join(CFG["gpudatadir"], "OUTPUTS_well_classified_" + xp_title)
            with open(tgt_file + "_TEST.pkl", 'wb') as f:
                pkl.dump(test_res, f)

        if hparams.test_on_train == "True":
            train_res = model.get_outputs_and_labels(DataLoader(dataset_train, batch_size=1,
                                                                collate_fn=collate_fn,
                                                                num_workers=0))
            with open(tgt_file + "_TRAIN.pkl", 'wb') as f:
                pkl.dump(train_res, f)


def load_datasets(hparams, CFG, splits):
    datasets = []
    if hparams.model_type.split("_")[-1] == "spe":
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "rep_type": hparams.rep_type,
            "agg_type": hparams.data_agg_type,
            "bag_type": hparams.bag_type,
            "subsample": 14000 if splits == ["TRAIN"] else 0
        }
        for split in splits:
            datasets.append(DiscriminativeSpecializedDataset(**common_hparams, split=split))
    else:
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "ppl_file": CFG["rep"][hparams.rep_type]["total"],
            "rep_type": hparams.rep_type,
            "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type + ".pkl",
            "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type + ".pkl",
            "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type + ".pkl",
            "agg_type": hparams.data_agg_type,
            "load": hparams.load_datasets,
            "subsample": 14000 if splits == ["TRAIN"] else 0
        }
        for split in splits:
            datasets.append(DiscriminativePolyvalentDataset(**common_hparams, split=split))
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=[0])
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--test_on_train", default=False)
    parser.add_argument("--well_classified", default=True)
    parser.add_argument("--input_type", type=str, default="bagTransformer")
    parser.add_argument("--model_type", type=str, default="SGDdisc_spe")
    parser.add_argument("--middle_size", type=int, default=50)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--b_size", type=int, default=64)
    hparams = parser.parse_args()
    main(hparams)
