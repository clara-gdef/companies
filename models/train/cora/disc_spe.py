import os
import torch

import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from data.datasets import DiscriminativeCoraDataset
from models.classes import InstanceClassifierDiscCora
from utils.models import collate_for_disc_spe_model_cora, get_model_params


def init(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return train(hparams)
    else:
        return train(hparams)


def main(hparams):
    xp_title = hparams.model_type + "_" + hparams.ft_type + "_" + hparams.input_type + "_bs" + str(
        hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, CFG, xp_title)
    print(hparams.auto_lr_find)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         logger=logger,
                         auto_lr_find=True
                         )
    datasets = load_datasets(hparams, CFG, ["TRAIN", "VALID"])
    dataset_train, dataset_valid = datasets[0], datasets[1]
    in_size, out_size = get_model_params(hparams, 300, len(dataset_train.track_rep))
    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model_cora,
                              num_workers=8, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model_cora,
                              num_workers=8)
    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "num_tracks": len(dataset_train.track_rep),
                 "input_type": hparams.input_type,
                 "ft_type": hparams.ft_type}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDiscCora(**arguments)
    print("Model Loaded.")
    if hparams.load_from_checkpoint:
        print("Loading from previous checkpoint...")
        model_name = xp_title
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)

        model_path = os.path.join(CFG['modeldir'], model_name)
        model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
        model.load_state_dict(torch.load(model_file)["state_dict"])
        print("Resuming training from checkpoint : " + model_file + ".")
    else:
        print("Starting training " + xp_title)

    trainer.fit(model.cuda(), train_loader, valid_loader)


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


def init_lightning(hparams, CFG, xp_title):
    model_path = os.path.join(CFG['modeldir'], xp_title)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_path, '{epoch:02d}'),
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )
    return logger, checkpoint_callback, early_stop_callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='pt')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--model_type", type=str, default="cora_disc_spe_std")
    parser.add_argument("--load_dataset", type=str, default="False")
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_from_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=49)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    main(hparams)
