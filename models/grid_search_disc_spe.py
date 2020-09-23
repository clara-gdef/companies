import argparse
import os
import pickle as pkl
import yaml
import ipdb
from models import train, eval
from utils import DotDict


def grid_search(hparams):
    dico = init_args(hparams)
    for bag_type in hparams.bag_types:
        test_results = {}
        for lr in [1e-8]:
            test_results[lr] = {}
            for b_size in [64, 512, 1024]:
                test_results[lr][b_size] = {}
                if hparams.input_type == "hadamard":
                    for mid_size in [200, 600, 1000]:
                        print("Grid Search for " + bag_type.upper() + " (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
                        dico['lr'] = lr
                        dico["b_size"] = b_size
                        dico["middle_size"] = mid_size
                        dico["bag_type"] = bag_type
                        arg = DotDict(dico)
                        if hparams.TRAIN == "True":
                            train.disc_spe.main(arg)
                        test_results[lr][b_size][mid_size] = eval.disc_spe.test(arg, CFG)
                else:
                    for wd in [0., .4, .8]:
                        print("Grid Search for (lr=" + str(lr) + ", b_size=" + str(b_size) + ", wd=" + str(
                            wd) + ")")
                        dico['lr'] = lr
                        dico["b_size"] = b_size
                        dico["middle_size"] = hparams.middle_size
                        dico["bag_type"] = bag_type
                        dico["wd"] = wd
                        arg = DotDict(dico)
                        if hparams.TRAIN == "True":
                            train.disc_spe.main(arg)
                        test_results[lr][b_size][wd] = eval.disc_spe.test(arg, CFG)
        res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_wd_disc_spe_" + bag_type + "_" + hparams.rep_type +
                                "_" + hparams.input_type)
        with open(res_path, "wb") as f:
            pkl.dump(test_results, f)


def init_args(hparams):
    dico = {'rep_type': hparams.rep_type,
            'gpus': hparams.gpus,
            'input_type': hparams.input_type,
            'load_dataset': True,
            'auto_lr_find': False,
            'data_agg_type': 'avg',
            'epochs': hparams.epochs,
            "wd": 0.,
            "load_from_checkpoint": False}
    print(hparams.epochs)
    return dico


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--middle_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--TRAIN", default=False)
    parser.add_argument("--bag_types", nargs='+', default=["cie", "dpt"])
    hparams = parser.parse_args()
    grid_search(hparams)
