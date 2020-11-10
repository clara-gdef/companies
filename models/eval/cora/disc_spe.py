import torch
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
from data.datasets import DiscriminativeCoraDataset
from models.classes import InstanceClassifierDiscCora
from utils.models import collate_for_disc_spe_model_cora, get_model_params, get_latest_model


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
    high_level = (hparams.high_level_classes == "True")
    if high_level:
        xp_title = hparams.model_type + "_HL_" + hparams.ft_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    else:
        xp_title = hparams.model_type + "_" + hparams.ft_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    logger = init_lightning(hparams, CFG, xp_title)
    print(hparams.auto_lr_find)
    trainer = pl.Trainer(gpus=hparams.gpus,
                         logger=logger,
                         )
    datasets = load_datasets(hparams, CFG, ["TEST"], high_level)
    dataset_test = datasets[0]
    in_size, out_size = get_model_params(hparams, 300, len(dataset_test.track_rep))
    test_loader = DataLoader(dataset_test, batch_size=1, collate_fn=collate_for_disc_spe_model_cora, num_workers=8,
                             shuffle=True)

    arguments = {'in_size': in_size,
                 'out_size': out_size,
                 'hparams': hparams,
                 'datadir': CFG["gpudatadir"],
                 'desc': xp_title,
                 "num_tracks": len(dataset_test.track_rep),
                 "input_type": hparams.input_type,
                 "ft_type": hparams.ft_type}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = InstanceClassifierDiscCora(**arguments)
    latest_model = get_latest_model(CFG["modeldir"], xp_title)
    print("Evaluating model " + latest_model)
    model.load_state_dict(torch.load(latest_model)["state_dict"])
    return trainer.test(model.cuda(), test_loader)


def load_datasets(hparams, CFG, splits, high_level):
    if high_level:
        bag_file = CFG["rep"]["cora"]["tracks"]
    else:
        bag_file = CFG["rep"]["cora"]["highlevelclasses"]
    datasets = []
    common_hparams = {
        "datadir": CFG["gpudatadir"],
        "bag_file": bag_file,
        "paper_file": CFG["rep"]["cora"]["papers"]["emb"],
        "ft_type": hparams.ft_type,
        "subsample": 0,
        "high_level": high_level,
        "load": hparams.load_dataset == "True"
    }
    for split in splits:
        datasets.append(DiscriminativeCoraDataset(**common_hparams, split=split))

    return datasets


def init_lightning(hparams, CFG, xp_title):
    logger = TensorBoardLogger(
        save_dir='./models/logs',
        name=xp_title)
    print("Logger initiated.")

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='pt')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--high_level_classes", type=str, default="True")
    parser.add_argument("--model_type", type=str, default="cora_disc_spe_adam")
    parser.add_argument("--load_dataset", type=str, default="False")
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_from_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=49)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    init(hparams)
