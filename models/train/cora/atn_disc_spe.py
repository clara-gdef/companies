import os
import torch

import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from models.classes import AtnInstanceClassifierDiscCora
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

    logger, checkpoint_callback, early_stop_callback = init_lightning(hparams, CFG, xp_title)
    call_back_list = [checkpoint_callback, early_stop_callback]
    if hparams.log_cm == "True":
        call_back_list.append(MyCallbacks())

    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         callbacks=call_back_list,
                         logger=logger
                         )
    datasets = load_datasets(hparams, CFG, ["TRAIN", "VALID"], hparams.high_level_classes == "True")
    dataset_train, dataset_valid = datasets[0], datasets[1]
    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model_cora,
                              num_workers=2, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model_cora,
                              num_workers=2)

    model = init_model(hparams, dataset_train, CFG["gpudatadir"], xp_title, AtnInstanceClassifierDiscCora)

    if hparams.load_from_checkpoint == "True":
        print("Loading from previous checkpoint...")
        model_name = xp_title
        if hparams.input_type == "hadamard":
            model_name += "/" + str(hparams.middle_size)

        model_path = os.path.join(CFG['modeldir'], model_name)
        model_file = os.path.join(model_path, "epoch=" + str(hparams.checkpoint) + ".ckpt")
        model.load_state_dict(torch.load(model_file)["state_dict"])
        print("Resuming training from checkpoint : " + model_file + ".")
    elif hparams.init == "True":
        print("Initializing weights with vanilla model...")
        if hparams.high_level_classes == "True":
            model_name = "cora_disc_spe_HL_eye_sgd_fs_matMul_bs16_1e-06_0.0/epoch=14.ckpt"
        else:
            model_name = "cora_disc_spe_rdn_init_fs_matMul_bs16_1e-08_0.0/epoch=188.ckpt"
        model_file = os.path.join(CFG['modeldir'], model_name)
        model.lin.weight = torch.nn.Parameter(torch.load(model_file)["state_dict"]["lin.weight"])
        model.lin.bias = torch.nn.Parameter(torch.load(model_file)["state_dict"]["lin.bias"])
        print("Initialized model weights with : " + model_file + ".")
    else:
        print("Starting training " + xp_title)


    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # #fig = lr_finder.plot(suggest=True)
    # #fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    #
    # # update hparams of the model
    # model.hparams.lr = new_lr
    # ipdb.set_trace()

    trainer.fit(model.cuda(), train_loader, valid_loader)


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


class MyCallbacks(Callback):
    def __init__(self):
        self.epoch = 0
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log_confusion_matrix(pl_module.valid_outputs, pl_module.valid_labels, "valid_ep_"+ str(self.epoch))
        print('Confusion Matrix logged for validation epoch ' + str(self.epoch))
        self.epoch += 1

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log_confusion_matrix(pl_module.train_outputs, pl_module.train_labels, "train_ep_"+ str(self.epoch))
        print('Confusion Matrix logged for train epoch ' + str(self.epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--init", type=str, default="False")
    parser.add_argument("--frozen", type=str, default="False")
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--model_type", type=str, default="atn_cora_disc_spe")
    parser.add_argument("--load_dataset", type=str, default="True")
    parser.add_argument("--high_level_classes", type=str, default="True")
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_from_checkpoint", type=str, default='False')
    parser.add_argument("--checkpoint", type=str, default=49)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--log_cm", type=str, default="True")
    parser.add_argument("--init_type", type=str, default="eye")
    hparams = parser.parse_args()
    init(hparams)
