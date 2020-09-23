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
#    for lr in [1e-4, 1e-6, 1e-8]:
    for lr in [1e-07, 1e-08]:
        test_results[lr] = {}
        for b_size in [512, 768, 1024]:
            test_results[lr][b_size] = {}
            if hparams.input_type == "hadamard":
                for mid_size in [50, 200, 600]:
                    print("Grid Search for (lr=" + str(lr) + ", b_size=" + str(b_size) + ", middle size=" + str(mid_size) + ")")
                    dico['lr'] = lr
                    dico["b_size"] = b_size
                    dico["middle_size"] = mid_size
                    arg = DotDict(dico)
                    if hparams.TRAIN == "True":
                        train.disc_poly.main(arg)
                    test_results[lr][b_size][mid_size] = eval.disc_poly.main(arg)
            else:
                print("Grid Search for (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
                dico['lr'] = lr
                dico["b_size"] = b_size
                dico["middle_size"] = hparams.middle_size
                arg = DotDict(dico)
                if hparams.TRAIN == "True":
                    train.disc_poly.main(arg)

                    test_results[lr][b_size] = eval.disc_poly.main(arg)
    res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_wd_topK_disc_poly_" + hparams.rep_type + "_" + hparams.input_type)
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
            "load_from_checkpoint": False,
            "checkpoint": 49,
            "wd": 0.0,
            "DEBUG": hparams.DEBUG,
            }
    return dico


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='sk')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--TRAIN", default=False)
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--middle_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    hparams = parser.parse_args()
    grid_search(hparams)
