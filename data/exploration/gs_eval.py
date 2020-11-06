import argparse
import os
import yaml
import pickle as pkl
import ipdb
import json
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    file_name = "EVAL_gs_cora_disc_spe_matMul"
    with ipdb.launch_ipdb_on_exception():
        res_path = os.path.join(CFG["gpudatadir"], file_name)
        with open(res_path, "rb") as f:
            test_results = pkl.load(f)
        ipdb.sey_trace()
        get_best_params(test_results, 'tracks', False, False)


def get_best_params(test_results, handle, weight_decay, mid_size):
    best_acc = 0
    best_f1 = 0
    best_acc_keys = None
    best_f1_keys = None
    if not weight_decay and not mid_size:
        for lr in test_results.keys():
            for bs in test_results[lr].keys():
                if test_results[lr][bs][0]["acc_" + handle] > best_acc:
                    best_acc_keys = (lr, bs)
                    best_acc = test_results[lr][bs][0]["acc_" + handle]
                if test_results[lr][bs][0]["f1_" + handle] > best_f1:
                    best_f1_keys = (lr, bs)
                    best_f1 = test_results[lr][bs][0]["f1_" + handle]
        print("Evaluated for lr= [" + str(test_results.keys()) + "], bs=[" +  str(test_results[lr].keys()) + "]")

    elif weight_decay and not mid_size:
        for lr in test_results.keys():
            for bs in test_results[lr].keys():
                for wd in test_results[lr][bs].keys():
                    if test_results[lr][bs][wd][0]["acc_" + handle] > best_acc:
                        best_acc_keys = (lr, bs, wd)
                        best_acc = test_results[lr][bs][wd][0]["acc_" + handle]
                    if test_results[lr][bs][wd][0]["f1_" + handle] > best_f1:
                        best_f1_keys = (lr, bs, wd)
                        best_f1 = test_results[lr][bs][wd][0]["f1_" + handle]
        print("Evaluated for lr= [" + str(test_results.keys()) + "], bs=[" +  str(test_results[lr].keys()) +
              "], wd=[" + str(test_results[lr][bs].keys()) + "]")

    else:
        for lr in test_results.keys():
            for bs in test_results[lr].keys():
                for ms in test_results[lr][bs].keys():
                    if test_results[lr][bs][ms][0]["acc_" + handle] > best_acc:
                        best_acc_keys = (lr, bs, ms)
                        best_acc = test_results[lr][bs][ms][0]["acc_" + handle]
                    if test_results[lr][bs][ms][0]["f1_" + handle] > best_f1:
                        best_f1_keys = (lr, bs, ms)
                        best_f1 = test_results[lr][bs][ms][0]["f1_" + handle]
        print("Evaluated for lr= [" + str(test_results.keys()) + "], bs=[" +  str(test_results[lr].keys()) +
              "], ms=[" + str(test_results[lr][bs].keys()) + "]")
    return best_acc_keys, best_f1_keys




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default='pt')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--model_type", type=str, default="cora_disc_spe_std")
    parser.add_argument("--load_dataset", type=str, default="False")
    parser.add_argument("--middle_size", type=int, default=250)
    parser.add_argument("--load_from_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=49)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    main(hparams)
