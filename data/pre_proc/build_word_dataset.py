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
            embedder = tmp.get_word_vector
        common_hparams = {
            "data_dir": CFG["gpudatadir"],
            "rep_type": hparams.rep_type,
            "embedder": embedder,
            "min_job_count": hparams.min_job_count,
            "max_job_count": 9,# covers 90% of the dataset
            "max_word_count": 126, # covers 90% of the dataset
            "load": False,
            "subsample":0
        }
        for split in ["TRAIN", "VALID", "TEST"]:
            WordDatasetSpe(**common_hparams, split=split)
        print("all datasets have been built!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--min_job_count", type=int, default=3)
    parser.add_argument("--load_dataset", default="True")
    parser.add_argument("--subsample", type=int, default=0)
    hparams = parser.parse_args()
    main(hparams)
