import argparse
import os
import pickle as pkl
import yaml
from models import train, eval
from utils import DotDict


def grid_search(hparams):
    dico = init_args(hparams)
    test_results = {}
    for lr in hparams.lr:
        test_results[lr] = {}
        for b_size in hparams.b_size:
            if hparams.input_type == "hadamard":
                test_results[lr][b_size] = {}
                for mid_size in [200, 600, 1000]:
                    print("Grid Search for CORA (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
                    dico['lr'] = lr
                    dico["b_size"] = b_size
                    dico["middle_size"] = mid_size
                    arg = DotDict(dico)
                    if hparams.TRAIN == "True":
                        train.cora.disc_spe.main(arg)
                    test_results[lr][b_size][mid_size] = eval.cora.disc_spe.main(arg)
            else:
                print("Grid Search for (lr=" + str(lr) + ", b_size=" + str(b_size) + ")")
                dico['lr'] = lr
                dico["b_size"] = b_size
                dico["middle_size"] = hparams.middle_size
                arg = DotDict(dico)
                if hparams.TRAIN == "True":
                    train.cora.disc_spe.main(arg)
                test_results[lr][b_size] = eval.cora.disc_spe.main(arg)
    res_path = os.path.join(CFG["gpudatadir"], "EVAL_gs_cora_disc_spe_" + hparams.rep_type +
                            "_" + hparams.input_type)
    with open(res_path, "wb") as f:
        pkl.dump(test_results, f)


def init_args(hparams):
    dico = {'ft_type': hparams.rep_type,
            'gpus': hparams.gpus,
            'input_type': hparams.input_type,
            'load_dataset': True,
            'auto_lr_find': False,
            "model_type": hparams.model_type,
            'data_agg_type': 'avg',
            'epochs': hparams.epochs,
            "wd": 0.,
            "subsample": 0,
            "DEBUG": hparams.DEBUG,
            "load_from_checkpoint": False}
    print(hparams.epochs)
    return dico


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='fs')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="cora_disc_spe")
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--middle_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--TRAIN", default="True")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--lr", nargs='+', default=[1e-4, 1e-6, 1e-8])
    parser.add_argument("--b_size", nargs='+', default=[64, 512, 16])
    hparams = parser.parse_args()
    grid_search(hparams)