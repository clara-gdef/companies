import os

import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import glob
import torch
from data.datasets import DiscriminativePolyvalentDataset
from models.classes import InstanceClassifierDisc
from utils.models import collate_for_disc_poly_model, get_model_params


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return test(hparams)
    else:
        return test(hparams)


def test(hparams):
    xp_title = "disc_poly_wd_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
        hparams.b_size)
    logger = init_lightning(hparams, CFG, xp_title)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         checkpoint_callback=None,
                         logger=logger,
                         )
    datasets = load_datasets(hparams, CFG, ["TRAIN"], True)
    dataset_train = datasets[0]
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

    dataset = load_datasets(hparams, CFG, ["TEST"], hparams.load_dataset)
    test_loader = DataLoader(dataset[0], batch_size=1, collate_fn=collate_for_disc_poly_model, num_workers=32)
    model_name = "disc_poly_wd/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type + "/" + \
                 str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
    if hparams.input_type == "hadamard":
        model_name += "/" + str(hparams.middle_size)
    model_path = os.path.join(CFG['modeldir'], model_name)
    model_files = glob.glob(os.path.join(model_path, "*"))
    latest_file = max(model_files, key=os.path.getctime)
    # latest_file = "/net/big/gainondefor/work/trained_models/companies/disc_poly_ce/ft/avg/matMul/512/1e-06/epoch=00.ckpt"
    print("Evaluating model: " + str(latest_file))
    model.load_state_dict(torch.load(latest_file)["state_dict"])
    model.eval()

    return trainer.test(test_dataloaders=test_loader, model=model.cuda())


def load_datasets(hparams, CFG, splits, load):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "ppl_file": CFG["rep"][hparams.rep_type]["total"],
        "rep_type": hparams.rep_type,
        "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type + ".pkl",
        "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type + ".pkl",
        "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type + ".pkl",
        "agg_type": hparams.data_agg_type,
        "load": load
    }
    for split in splits:
        datasets.append(DiscriminativePolyvalentDataset(**common_hparams, split=split))

    return datasets


def init_lightning(hparams, CFG, xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='sk')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--b_size", type=int, default=512)
    parser.add_argument("--wd", type=float, default=.4)
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    hparams = parser.parse_args()
    main(hparams)

