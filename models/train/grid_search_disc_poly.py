import argparse
import yaml
from models.train import disc_poly
from utils import DotDict


def grid_search(hparams):
    dico = init_args(hparams)

    for lr in [1e-4, 1e-6, 1e-8]:
        for b_size in [16, 64, 512]:
            print("Grid Search for couple (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
            dico['lr'] = lr
            dico["b_size"] = b_size
            arg = DotDict(dico)
            disc_poly.main(arg)


def init_args(hparams):
    dico = {'rep_type': hparams.rep_type,
            'gpus': hparams.gpus,
            'input_type': hparams.input_type,
            'load_dataset': True,
            'auto_lr_find': False,
            'data_agg_type': 'avg',
            'epochs': 50}
    return dico


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--epochs", type=int, default=100)
    hparams = parser.parse_args()
    grid_search(hparams)
