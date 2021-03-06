import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from data.datasets import GenerativePolyvalentDataset
from models.classes.InstanceClassifierGen import InstanceClassifierGen
from utils.models import collate_for_gen_poly_model


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        train(hparams)


def train(hparams):
    xp_title = "gen_poly_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
        hparams.b_size)
    logger, checkpoint_callback, early_stop_callback = init_lightning(xp_title)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback,
                         logger=logger,
                         auto_lr_find=hparams.auto_lr_find
                         )
    datasets = load_datasets(hparams, ["TRAIN", "VALID"], hparams.load_dataset)
    dataset_train, dataset_valid = datasets[0], datasets[1]
    in_size, out_size = get_model_params(len(dataset_train))

    train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_gen_poly_model,
                              num_workers=32)
    valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_gen_poly_model,
                              num_workers=32)
    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'dataset': dataset_train,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierGen(**arguments)
    print("Model Loaded.")
    print("Starting training...")
    ipdb.set_trace()
    trainer.fit(model, train_loader, valid_loader)


def load_datasets(hparams, splits, load):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "ppl_file": CFG["rep"][hparams.rep_type]["total"],
        "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type + ".pkl",
        "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type + ".pkl",
        "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type + ".pkl",
        "rep_type": hparams.rep_type,
        "load": load
    }
    for split in splits:
        datasets.append(GenerativePolyvalentDataset(**common_hparams, split=split))
    return datasets


def get_model_params(rep_dim):
    out_size = 1
    if hparams.input_type == "hadamard" or hparams.input_type == "concat":
        in_size = rep_dim * 2
    elif hparams.input_type == "matMul":
        in_size = rep_dim
    else:
        raise Exception("Wrong input data specified: " + str(hparams.input_type))

    return in_size, out_size


def init_lightning(xp_title):
    model_path = os.path.join(CFG['modeldir'], "gen_poly/" + hparams.rep_type + "/" + hparams.data_agg_type)

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
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode='min'
    )
    return logger, checkpoint_callback, early_stop_callback


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=[0])
    parser.add_argument("--b_size", type=int, default=2)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    hparams = parser.parse_args()
    main(hparams)
