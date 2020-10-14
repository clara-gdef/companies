import glob
import os
import ipdb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml
import torch
from data.datasets import JobsDatasetPoly
from models.classes.AtnInstanceClassifierDisc import AtnInstanceClassifierDisc
from utils.models import collate_for_attn_disc_poly_model, get_model_params


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if hparams.DEBUG:
        with ipdb.launch_ipdb_on_exception():
            return evaluate(hparams)
    else:
        return evaluate(hparams)


def evaluate(hparams):
    xp_title = hparams.model_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_" + \
               str(hparams.b_size) + "_" + str(hparams.lr) + '_' + str(hparams.wd)
    if hparams.input_type == "hadamard":
        xp_title += "_" + str(hparams.middle_size)

    logger = init_lightning(xp_title)
    trainer = pl.Trainer(gpus=[hparams.gpus],
                         logger=logger
                         )
    dataset_test = load_datasets(hparams, ["TEST"], hparams.load_dataset)[0]

    in_size, out_size = get_model_params(hparams, dataset_test.rep_dim, len(dataset_test.bag_rep))
    test_loader = DataLoader(dataset_test, batch_size=hparams.b_size, collate_fn=collate_for_attn_disc_poly_model,
                              num_workers=0, shuffle=True)
    print("Dataloaders initiated.")

    print("Loading previously saved classifier...")
    model_name = "disc_poly_wd/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type +\
                 "/768/1e-08/0.7/epoch=" + hparams.prev_model_ep +".ckpt"
    weights = torch.load(os.path.join(CFG['modeldir'], model_name))["state_dict"]

    arguments = {'dim_size': 300,
                 'in_size': in_size,
                 'out_size': out_size,
                 "num_cie": dataset_test.num_cie,
                 "num_clus": dataset_test.num_clus,
                 "num_dpt": dataset_test.num_dpt,
                 'hparams': hparams,
                 'desc': xp_title,
                 "middle_size": hparams.middle_size,
                 "fixed_weights": weights}

    print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
    model = AtnInstanceClassifierDisc(**arguments)
    print("Model Loaded.")
    model_name = hparams.model_type + "/" + hparams.rep_type + "/" + hparams.data_agg_type + "/" + hparams.input_type + "/" + \
                 str(hparams.b_size) + "/" + str(hparams.lr) + "/" + str(hparams.wd)
    if hparams.input_type == "hadamard":
        model_name += "/" + str(hparams.middle_size)
    model_path = os.path.join(CFG['modeldir'], model_name)
    model_files = glob.glob(os.path.join(model_path, "*"))
    latest_file = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_file)["state_dict"])

    return trainer.test(test_dataloaders=test_loader, model=model)


def load_datasets(hparams, splits, load):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "ppl_file": CFG["rep"][hparams.rep_type]["total"],
        "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type + ".pkl",
        "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type + ".pkl",
        "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type + ".pkl",
        "load": load
    }
    for split in splits:
        datasets.append(JobsDatasetPoly(**common_hparams, split=split))
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
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=512)
    parser.add_argument("--middle_size", type=int, default=20)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", default="False")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--prev_model_ep", type=str, default='99')
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="frozenAtn_disc_poly")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    main(hparams)
