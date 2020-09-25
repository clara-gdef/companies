import os

import ipdb
import argparse

from torch.utils.data import DataLoader
import yaml
import torch
import pickle as pkl
import glob
from data.datasets import JobsDataset, DiscriminativeSpecializedDataset
from models.classes import InstanceClassifierDisc
from utils.models import collate_for_jobs, get_model_params


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        xp_title = hparams.model_type + "_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size)

        dataset_train, dataset_jobs = load_datasets(hparams, CFG)
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

        test_loader = DataLoader(dataset_jobs, batch_size=1, collate_fn=collate_for_jobs, num_workers=0)
        model_name = hparams.model_type + "/" + hparams.bag_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + \
                     "/" + hparams.input_type + "/" + str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)

        if hparams.input_type != "b4Training":
            model_path = os.path.join(CFG['modeldir'], model_name)
            model_files = glob.glob(os.path.join(model_path, "*"))
            latest_file = max(model_files, key=os.path.getctime)
            print("Evaluating model: " + str(latest_file))
            model.load_state_dict(torch.load(latest_file)["state_dict"])

        outputs_for_jobs = model.get_jobs_outputs(test_loader)
        tgt_file = os.path.join(CFG["gpudatadir"], "jobs_outputs_" + xp_title + ".pkl")
        with open(tgt_file, 'wb') as f:
            torch.save(outputs_for_jobs, f)



def load_datasets(hparams, CFG):
    if hparams.model_type.split("_")[-1] == "spe":
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "rep_type": hparams.rep_type,
            "agg_type": hparams.data_agg_type,
            "bag_type": hparams.bag_type,
            "subsample": 0,
            "split": "TRAIN"
        }
        dataset_init = (DiscriminativeSpecializedDataset(**common_hparams))

        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "rep_type": hparams.rep_type,
            "bag_type": hparams.bag_type,
            "bag_rep": dataset_init.bag_rep
        }
        dataset_jobs = JobsDataset(**common_hparams)
    return dataset_init, dataset_jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=[0])
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--test_on_train", default=True)
    parser.add_argument("--well_classified", default=True)
    parser.add_argument("--input_type", type=str, default="bagTransformer")
    parser.add_argument("--model_type", type=str, default="disc_spe")
    parser.add_argument("--middle_size", type=int, default=50)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--b_size", type=int, default=64)
    hparams = parser.parse_args()
    main(hparams)
