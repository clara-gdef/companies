import os
import torch

import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from data.datasets import DiscriminativeSpecializedDataset
from models.classes import InstanceClassifierDisc
from utils.models import collate_for_disc_spe_model, get_model_params


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
    xp_title = hparams.model_type + "_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
        hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, CFG, xp_title)

    call_back_list = [checkpoint_callback, early_stop_callback]
    if hparams.log_cm == "True":
        call_back_list.append(MyCallbacks())

    print(hparams.auto_lr_find)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         callbacks=call_back_list,
                         logger=logger,
                         auto_lr_find=hparams.auto_lr_find
                         )
    datasets = load_datasets(hparams, CFG, ["TRAIN", "VALID"])
    dataset_train, dataset_valid = datasets[0], datasets[1]
    in_size, out_size = get_model_params(hparams, dataset_train.rep_dim, dataset_train.get_num_bag())
    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                              num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                              num_workers=0)
    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'dataset': dataset_train,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "bag_type": hparams.bag_type,
                 "wd": hparams.wd,
                 "middle_size": hparams.middle_size}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDisc(**arguments)
    print("Model Loaded.")
    if hparams.load_from_checkpoint:
        print("Loading from previous checkpoint...")
        model_name = hparams.model_type + "/" + hparams.bag_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + \
                     "/" + hparams.input_type + "/" + str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(
            hparams.wd)
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)

        model_path = os.path.join(CFG['modeldir'], model_name)
        model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
        model.load_state_dict(torch.load(model_file)["state_dict"])
        print("Resuming training from checkpoint : " + model_file + ".")
    else:
        print("Starting training " + xp_title)

    if hparams.auto_lr_find == "True":
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
        # Results can be found in
        print(lr_finder.results)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.lr = new_lr
        ipdb.set_trace()

    trainer.fit(model, train_loader, valid_loader)


def load_datasets(hparams, CFG, splits):
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


def init_lightning(hparams, CFG, xp_title):
    model_name = hparams.model_type + "/" + hparams.bag_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + \
                 "/" + hparams.input_type + "/" + str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
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
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0000,
        patience=10,
        verbose=True,
        mode='min'
    )
    return logger, checkpoint_callback, early_stop_callback

class MyCallbacks(Callback):
    def __init__(self):
        self.epoch = 0

    def on_sanity_check_end(self, trainer, pl_module):
        pl_module.log_confusion_matrix(torch.stack(pl_module.valid_outputs),
                                       torch.stack(pl_module.valid_labels),
                                       "sanity_check")
        print('Confusion Matrix logged for sanity check')

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log_confusion_matrix(torch.stack(pl_module.valid_outputs),
                                       torch.stack(pl_module.valid_labels),
                                       "valid_ep_" + str(self.epoch))
        print('Confusion Matrix logged for validation epoch ' + str(self.epoch))

    def on_train_epoch_end(self, trainer, pl_module, tmp):
        pl_module.log_confusion_matrix(torch.stack(pl_module.train_outputs),
                                       torch.stack(pl_module.train_labels),
                                       "train_end_ep_" + str(self.epoch))
        print('Confusion Matrix logged for train end epoch ' + str(self.epoch))
        self.epoch += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--standardized", type=str, default="True")
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--model_type", type=str, default="disc_spe_std")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_from_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=49)
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--log_cm", type=str, default="False")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--auto_lr_find", type=str, default="False")
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
