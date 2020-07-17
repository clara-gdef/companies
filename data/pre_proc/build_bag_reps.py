import yaml
import os
import ipdb
import pickle as pkl
from tqdm import tqdm
import torch


def main():
    dd = cfg["datadir"]
    print("Loading data...")
    with open(os.path.join(dd, 'lookup_ppl.pkl'), 'rb') as f:
        ppl_dict = pkl.load(f)
    with open(os.path.join(dd, 'lookup_cie.pkl'), 'rb') as f:
        cie_dict = pkl.load(f)
    with open(os.path.join(dd, 'lookup_clus.pkl'), 'rb') as f:
        clus_dict = pkl.load(f)
    with open(os.path.join(dd, 'lookup_dpt.pkl'), 'rb') as f:
        dpt_dict = pkl.load(f)
    print("Data loaded")
    with ipdb.launch_ipdb_on_exception():
        for (agg_name, agg_func) in [("avg", torch.mean), ("max", torch.max), ("sum", torch.sum)]:
            build_and_save_rep_for_bag("cie", cie_dict, ppl_dict, agg_name, agg_func)
            build_and_save_rep_for_bag("clus", clus_dict, ppl_dict, agg_name, agg_func)
            build_and_save_rep_for_bag("dpt", dpt_dict, ppl_dict, agg_name, agg_func)


def build_and_save_rep_for_bag(bag_name, bag_dict, ppl_dict, agg_name, agg_func):
    bag_reps = dict()
    for bag_num, bag_name in tqdm(bag_dict.items(),
                                  desc="Building rep of " + bag_name.upper() + " for agg " + agg_name.upper() + "..."):
        ppl_ft_emb = [ppl_dict[i]["ft"] for i in ppl_dict if ppl_dict[i][bag_name + "_name"] == bag_name]
        ppl_skills = [ppl_dict[i]["sk"] for i in ppl_dict if ppl_dict[i][bag_name + "_name"] == bag_name]
        if bag_num not in bag_reps.keys():
            bag_reps[bag_num] = dict()
        if agg_name == "max":
            bag_reps[bag_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0)[0].unsqueeze(0)
            bag_reps[bag_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)[0]
        else:
            bag_reps[bag_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0).unsqueeze(0)
            bag_reps[bag_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)
    with open(os.path.join(cfg["datadir"], bag_name + "_rep_" + agg_name + ".pkl"), "wb") as f:
        pkl.dump(bag_reps, f)


if __name__ == "__main__":
    global cfg
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    main()
