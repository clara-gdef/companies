import os

import torch
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import yaml
from models.classes import AtnInstanceClassifierDiscCora
from utils.models import get_latest_model
from utils.cora import load_datasets, collate_for_disc_spe_model_cora, init_model, xp_title_from_params


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return main(hparams)
    else:
        return main(hparams)


def main(hparams):
    xp_title = xp_title_from_params(hparams)

    logger = init_lightning(hparams, CFG, xp_title)
    print(hparams.auto_lr_find)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         logger=logger,
                         )
    # TODO : REMOVE TRAIN AND REPLACE BY TEST, THIS IS FOR DBEUG
    datasets = load_datasets(hparams, CFG, ["TEST"],  hparams.high_level_classes == "True")
    dataset_test = datasets[0]
    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_disc_spe_model_cora, num_workers=8,
                             shuffle=True)

    model = init_model(hparams, dataset_test, CFG["gpudatadir"], xp_title, AtnInstanceClassifierDiscCora)

    if hparams.load_from_checkpoint == "True":
        model_name = xp_title
        model_path = os.path.join(CFG['modeldir'], model_name)
        latest_model = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
    else:
        latest_model = get_latest_model(CFG["modeldir"], xp_title)
    print("Evaluating model " + latest_model)
    model.load_state_dict(torch.load(latest_model)["state_dict"])
    return trainer.test(model.cuda(), test_loader)


def init_lightning(hparams, CFG, xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--init", type=str, default="False")
    parser.add_argument("--frozen", type=str, default="False")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--log_cm", type=str, default="False")
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--model_type", type=str, default="atn_cora_disc_spe")
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_from_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=49)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--init_type", type=str, default="eye")
    parser.add_argument("--high_level_classes", type=str, default="True")
    hparams = parser.parse_args()
    init(hparams)
