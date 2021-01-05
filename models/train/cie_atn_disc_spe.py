import glob
import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import JobsDatasetSpe
from models.classes.AtnInstanceClassifierDisc import AtnInstanceClassifierDisc
from utils.models import collate_for_attn_disc_spe_model, get_model_params


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
    xp_title = make_xp_title(hparams)

    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, xp_title)
    call_back_list = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(gpus=[hparams.gpus],
                         max_epochs=hparams.epochs,
                         callbacks=call_back_list,
                         logger=logger
                         )
    datasets = load_datasets(hparams, ["TRAIN", "VALID"], hparams.load_dataset)
    dataset_train, dataset_valid = datasets[0], datasets[1]

    in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, len(dataset_train.bag_reps))
    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_attn_disc_spe_model,
                              num_workers=4, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_attn_disc_spe_model,
                              num_workers=4)
    print("Dataloaders initiated.")
    arguments = {'dim_size': 300,
                 'in_size': in_size,
                 'out_size': out_size,
                 "num_cie": dataset_train.num_bags,
                 "num_clus": 0,
                 "num_dpt": 0,
                 'hparams': hparams,
                 'desc': xp_title,
                 "data_dir": CFG["gpudatadir"],
                 "frozen": hparams.frozen,
                 "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = AtnInstanceClassifierDisc(**arguments)
    print("Model Loaded.")
    if hparams.load_from_checkpoint == "True":
        print("Loading from previous checkpoint...")
        model_name = hparams.model_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type + "/" + \
                     str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)
        model_path = os.path.join(CFG['modeldir'], model_name)
        model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
        model.load_state_dict(torch.load(model_file)["state_dict"])
        print("Resuming training from checkpoint : " + model_file + ".")
    elif hparams.init_weights == "True":
        print("Initializing class prediction weights...")
        model_name = 'disc_spe_std/cie/ft/avg/matMul/16/1e-06/0.0/epoch=02.ckpt'
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)
        model_path = os.path.join(CFG['modeldir'], model_name)
        model.lin.weight = torch.nn.Parameter(torch.load(model_path)["state_dict"]["lin.weight"])
        model.lin.bias = torch.nn.Parameter(torch.load(model_path)["state_dict"]["lin.bias"])
        print("Prediction layer initiated.")

    if hparams.auto_lr_find == "True":
        print("looking for best lr...")
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

        # Results can be found in
        print(lr_finder.results)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.lr = new_lr
        ipdb.set_trace()
    print("Starting training for " + xp_title + "...")

    trainer.fit(model.cuda(), train_loader, valid_loader)


def load_datasets(hparams, splits, load):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "bag_file": CFG["rep"]["cie"] + hparams.data_agg_type,
        "bag_type": "cie",
        "load": load,
        "standardized": True,
        "subsample": 0
    }
    for split in splits:
        datasets.append(JobsDatasetSpe(**common_hparams, split=split))

    return datasets


def init_lightning(hparams, xp_title):
    model_name = ""
    if hparams.frozen == "True":
        model_name += "frozen_"
    if hparams.init_weights == "True":
        model_name += "init_"
    model_name += hparams.model_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type + "/" + \
                 str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
    if hparams.input_type == "hadamard":
        model_name += "/" + str(hparams.middle_size)

    model_path = os.path.join(CFG['modeldir'], model_name)

    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_path, '{epoch:02d}'),
        save_top_k=1,
        verbose=True,
        monitor='valid_acc',
        mode='max',
        prefix=''
    )
    print("callback initiated.")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.000,
        patience=10,
        verbose=False,
        mode='min'
    )
    print("early stopping procedure initiated.")

    return logger, checkpoint_callback, early_stop_callback


def make_xp_title(hparams):
    xp_title = ""
    if hparams.frozen == "True":
        xp_title += "frozen_"
    if hparams.init_weights == "True":
        xp_title += "init_"
    xp_title += hparams.model_type + "_" + hparams.bag_type + "_" + hparams.rep_type + "_" +\
               hparams.data_agg_type + "_" + hparams.input_type + "_" + \
               str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    if hparams.input_type == "hadamard":
        xp_title += "_" + str(hparams.middle_size)
    print("xp_title = " + xp_title)
    return xp_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--middle_size", type=int, default=20)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--init_weights", default="True")
    parser.add_argument("--frozen", default="True")
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--load_from_checkpoint", default="False")
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--DEBUG", type=str, default="False")
    parser.add_argument("--model_type", type=str, default="atn_disc_spe")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
