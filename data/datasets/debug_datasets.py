import argparse
from operator import itemgetter

import ipdb
import yaml
from tqdm import tqdm
import pickle as pkl

from data.datasets import JobsDatasetPoly, DiscriminativePolyvalentDataset


def main(hparams):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    splits = ["VALID", "TEST", "TRAIN"]
    with ipdb.launch_ipdb_on_exception():
        for split in splits:
            print("SPLIT = " + split)
            common_hparams = {
                "data_dir": CFG["gpudatadir"],
                "ppl_file": CFG["rep"][hparams.rep_type]["total"],
                "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type,
                "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type,
                "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type,
                "load": False,
                "standardized": True
            }
            if hparams.standardized == "True":
                print("Loading standardized datasets...")
                common_hparams["standardized"] = True

            dataset_jobs = JobsDatasetPoly(**common_hparams, split=split)

            common_hparams = {
                "data_dir": CFG["gpudatadir"],
                "ppl_file": CFG["rep"][hparams.rep_type]["total"],
                "rep_type": hparams.rep_type,
                "cie_reps_file": CFG["rep"]["cie"] + hparams.data_agg_type,
                "clus_reps_file": CFG["rep"]["clus"] + hparams.data_agg_type,
                "dpt_reps_file": CFG["rep"]["dpt"] + hparams.data_agg_type,
                "agg_type": hparams.data_agg_type,
                "load": False,
                "subsample": 0,
                "standardized": True
            }
            if hparams.standardized == "True":
                print("Loading standardized datasets...")
                common_hparams["standardized"] = True

            dataset_profile = DiscriminativePolyvalentDataset(**common_hparams, split=split)

            assert len(dataset_jobs.tuples) == len(dataset_profile.tuples)
            assert dataset_jobs.num_cie == dataset_profile.num_cie
            assert dataset_jobs.num_clus == dataset_profile.num_clus
            assert dataset_jobs.num_dpt == dataset_profile.num_dpt
            sorted_jobs = sorted(dataset_jobs.tuples, key=itemgetter('id'))
            sorted_prof = sorted(dataset_profile.tuples, key=itemgetter('id'))
            for job, prof in tqdm(zip(sorted_jobs, sorted_prof), total=len(dataset_profile.tuples)):
                assert job["cie"] == prof["cie"]
                assert job["clus"] == prof["clus"]
                assert job["dpt"] == prof["dpt"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--standardized", type=str, default="False")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--load_dataset", default="True")
    hparams = parser.parse_args()
    main(hparams)
