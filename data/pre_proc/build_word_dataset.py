import os
import ipdb
import argparse
import yaml
import fastText
from data.datasets import WordDatasetSpe


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        print("Loading word vectors...")
        if hparams.rep_type == "ft":
            tmp = fastText.load_model(os.path.join(CFG["modeldir"], "ft_fs.bin"))
            embedder = tmp.get_word_vector()
        datasets = []
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "rep_type": hparams.rep_type,
            "embedder": embedder,
            "min_job_count": hparams.min_job_count,
            "max_word_count": 126, # covers 90% of the dataset
            "load": False,
            "subsample":0
        }
        for split in ["TRAIN", "VALID", "TEST"]:
            datasets.append(WordDatasetSpe(**common_hparams, split=split))
        print("all datasets have been built!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=768)
    parser.add_argument("--middle_size", type=int, default=20)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--init_weights", default="True")
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--load_from_checkpoint", default=False)
    parser.add_argument("--standardized", type=str, default="True")
    parser.add_argument("--checkpoint", type=int, default=45)
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--DEBUG", type=bool, default=False)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="atn_disc_poly_std")
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--wd", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    main(hparams)
