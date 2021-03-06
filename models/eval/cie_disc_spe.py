import os

import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
import glob
from data.datasets import DiscriminativeSpecializedDataset
from models.classes import InstanceClassifierDisc
from utils.models import collate_for_disc_spe_model, get_model_params


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return main(hparams, CFG)
    else:
        return main(hparams, CFG)


def main(hparams, CFG):
    xp_title = hparams.model_type + "_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" \
               + hparams.input_type + "_bs" + str(
        hparams.b_size)
    logger = init_lightning(xp_title)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         checkpoint_callback=None,
                         logger=logger,
                         )
    datasets = load_datasets(hparams, CFG, ["TRAIN"])
    dataset_train = datasets[0]
    in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, len(dataset_train.bag_rep))

    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'dataset': dataset_train,
                 "bag_type": hparams.bag_type,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "wd": hparams.wd,
                 "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDisc(**arguments)
    print("Model Loaded.")

    model.eval()

    dataset = load_datasets(hparams, CFG, ["TEST"])
    test_loader = DataLoader(dataset[0], batch_size=1, collate_fn=collate_for_disc_spe_model, num_workers=32)
    model_name = hparams.model_type + "/" + hparams.bag_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + \
                 "/" + hparams.input_type + "/" + str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
    if hparams.input_type == "hadamard":
        model_name += "/" + str(hparams.middle_size)
    model_path = os.path.join(CFG['modeldir'], model_name)
    model_files = glob.glob(os.path.join(model_path, "*"))
    latest_file = max(model_files, key=os.path.getctime)
    print("Evaluating model: " + str(latest_file))
    model.load_state_dict(torch.load(latest_file)["state_dict"])
    return trainer.test(test_dataloaders=test_loader, model=model)


def load_datasets(hparams, CFG,  splits):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "rep_type": hparams.rep_type,
        "agg_type": hparams.data_agg_type,
        "bag_type": hparams.bag_type,
        "subsample": 0,
        "standardized": False
    }
    if hparams.standardized == "True":
        print("Loading standardized datasets...")
        common_hparams["standardized"] = True
    for split in splits:
        datasets.append(DiscriminativeSpecializedDataset(**common_hparams, split=split))

    return datasets


def init_lightning(xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=[0])
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--standardized", type=str, default="True")
    parser.add_argument("--model_type", type=str, default="disc_spe_std")
    parser.add_argument("--middle_size", type=int, default=200)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--b_size", type=int, default=16)
    hparams = parser.parse_args()
    init(hparams)
