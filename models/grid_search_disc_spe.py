import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train, eval
from utils import DotDict


def grid_search(hparams):
    test_results = {}
    dico = init_args(hparams)
    for bag_type in ["cie", "clus", "dpt"]:
        test_results[bag_type] = {}
        for lr in [1e-4, 1e-6, 1e-8]:
            test_results[bag_type][lr] = {}
            for b_size in [16, 64, 512]:
                print("Grid Search for " + bag_type.upper() + " (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
                dico['lr'] = lr
                dico["b_size"] = b_size
                dico["bag_type"] = bag_type
                arg = DotDict(dico)
                train.disc_spe.main(arg)
                test_results[bag_type][lr][b_size] = eval.disc_spe.test(arg, CFG)
        res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_all_disc_spe_" + bag_type + "_" + hparams.rep_type + "_" + hparams.input_type)
        with open(res_path, "wb") as f:
            pkl.dump(test_results, f)


def init_args(hparams):
    dico = {'rep_type': hparams.rep_type,
            'gpus': hparams.gpus,
            'input_type': hparams.input_type,
            'load_dataset': True,
            'auto_lr_find': False,
            'data_agg_type': 'avg',
            'epochs': 50,
            "middle_size": hparams.middle_size}
    return dico


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--input_type", type=str, default="bagTransformer")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--middle_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    hparams = parser.parse_args()
    grid_search(hparams)
