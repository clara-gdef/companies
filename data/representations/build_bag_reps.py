import os
import pickle as pkl

import yaml
import ipdb
from tqdm import tqdm
import torch


def main():
    d_d = CFG["gpudatadir"]
    with ipdb.launch_ipdb_on_exception():
        print("Loading data...")
        with open(os.path.join(d_d, 'lookup_ppl.pkl'), 'rb') as f_name:
            ppl_dict = pkl.load(f_name)
        print("People loaded.")
        with open(os.path.join(d_d, 'lookup_cie.pkl'), 'rb') as f_name:
            cie_dict = pkl.load(f_name)
        print("Companies loaded.")
        with open(os.path.join(d_d, 'lookup_clus.pkl'), 'rb') as f_name:
            clus_dict = pkl.load(f_name)
        print("Clusters loaded.")
        with open(os.path.join(d_d, 'lookup_dpt.pkl'), 'rb') as f_name:
            dpt_dict = pkl.load(f_name)
        print("Departments loaded.")
        print("Data loaded")
        for (agg_name, agg_func) in [("avg", torch.mean), ("max", torch.max), ("sum", torch.sum)]:
            build_and_save_rep_for_bag("cie", cie_dict, ppl_dict, agg_name, agg_func)
            build_and_save_rep_for_bag("clus", clus_dict, ppl_dict, agg_name, agg_func)
            build_and_save_rep_for_bag("dpt", dpt_dict, ppl_dict, agg_name, agg_func)


def build_and_save_rep_for_bag(bag_type, bag_dict, ppl_dict, agg_name, agg_func):
    bag_reps = dict()
    for bag_num, bag_name in tqdm(bag_dict.items(),
                                  desc="Building rep of " + bag_type.upper()
                                  + " for agg " + agg_name.upper() + "..."):
        ppl_ft_emb = [ppl_dict[i]["ft"] for i in ppl_dict
                      if ppl_dict[i][bag_type + "_name"] == bag_name]
        # ppl_skills = [ppl_dict[i]["sk"] for i in ppl_dict
        #               if ppl_dict[i][bag_type + "_name"] == bag_name]
        if bag_num not in bag_reps.keys():
            bag_reps[bag_num] = dict()
        if agg_name == "max":
            bag_reps[bag_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0)[0].unsqueeze(0)
            # bag_reps[bag_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)[0]
        else:
            bag_reps[bag_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0).unsqueeze(0)
            # bag_reps[bag_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)
    with open(os.path.join(CFG["gpudatadir"], bag_type + "_rep_" + agg_name + "_standardized.pkl"), "wb") as f_name:
        pkl.dump(bag_reps, f_name)


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main()
