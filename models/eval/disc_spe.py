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


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        test(hparams, CFG)


def test(hparams, CFG):
    xp_title = "disc_spe_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
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
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDisc(**arguments)
    print("Model Loaded.")

    model.eval()

    dataset = load_datasets(hparams, CFG, ["TEST"])
    test_loader = DataLoader(dataset[0], batch_size=1, collate_fn=collate_for_disc_spe_model, num_workers=32)
    model_path = os.path.join(CFG['modeldir'], "disc_spe/" + hparams.bag_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type + "/" +
                              str(hparams.b_size) + "/" + str(hparams.lr))
    model_files = glob.glob(os.path.join(model_path, "*"))
    latest_file = max(model_files, key=os.path.getctime)
    print("Evaluating model: " + str(latest_file))
    model.load_state_dict(torch.load(latest_file)["state_dict"])
    trainer.test(test_dataloaders=test_loader, model=model)


def load_datasets(hparams, CFG,  splits):
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


def init_lightning(xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")
    return logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='sk')
    parser.add_argument("--gpus", type=int, default=[0])
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--middle_size", type=int, default=50)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--b_size", type=int, default=64)
    hparams = parser.parse_args()
    main(hparams)
