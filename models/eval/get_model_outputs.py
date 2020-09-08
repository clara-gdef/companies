import os

import ipdb
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
import glob
import torch
import pickle as pkl
from models.classes import InstanceClassifierDisc
from models.eval.disc_poly import load_datasets, get_model_params
from utils.models import collate_for_disc_poly_model


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with ipdb.launch_ipdb_on_exception():
        xp_title = "disc_poly_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size)
        datasets = load_datasets(hparams, CFG, ["TRAIN"], True)
        dataset_train = datasets[0]
        in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, len(dataset_train.bag_rep))

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

        dataset = load_datasets(hparams, CFG, ["TEST"], hparams.load_dataset)
        test_loader = DataLoader(dataset[0], batch_size=1, collate_fn=collate_for_disc_poly_model, num_workers=32)
        model_path = os.path.join(CFG['modeldir'], "disc_poly_w_init/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type + "/" +
                                  str(hparams.b_size) + "/" + str(hparams.lr))
        model_files = glob.glob(os.path.join(model_path, "*"))
        latest_file = max(model_files, key=os.path.getctime)
        print("Evaluating model: " + str(latest_file))
        model.load_state_dict(torch.load(latest_file)["state_dict"])
        model.eval()
        outputs = model.get_outputs_and_labels(test_loader)
        tgt_file = os.path.join(CFG["gpudatadir"], "OUTPUTS_" + model.description + ".pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(outputs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--b_size", type=int, default=512)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    hparams = parser.parse_args()
    main(hparams)

