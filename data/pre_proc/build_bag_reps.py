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
            cie_reps = dict()
            for cie_num, cie_name in tqdm(cie_dict.items(), desc="Building rep of CIE for agg " + agg_name + "..."):
                ppl_ft_emb = [ppl_dict[i]["ft"] for i in ppl_dict if ppl_dict[i]["cie_name"] == cie_name]
                ppl_skills = [ppl_dict[i]["sk"] for i in ppl_dict if ppl_dict[i]["cie_name"] == cie_name]
                if cie_num not in cie_reps.keys():
                    cie_reps[cie_num] = dict()
                cie_reps[cie_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0).unsqueeze(0)
                cie_reps[cie_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)
            with open(os.path.join(dd, "cie_rep_" + agg_name + ".pkl"), "wb") as f:
                pkl.dump(cie_reps, f)
            clus_reps = dict()
            for clus_num, clus_name in tqdm(clus_dict.items(), desc="Building rep of CLUS for agg " + agg_name + "..."):
                ppl_ft_emb = [ppl_dict[i]["ft"] for i in ppl_dict if ppl_dict[i]["clus_name"] == clus_name]
                ppl_skills = [ppl_dict[i]["sk"] for i in ppl_dict if ppl_dict[i]["clus_name"] == clus_name]
                if clus_num not in clus_reps.keys():
                    clus_reps[clus_num] = dict()
                clus_reps[clus_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0)
                clus_reps[clus_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)
            with open(os.path.join(dd, "clus_rep_" + agg_name + ".pkl"), "wb") as f:
                pkl.dump(clus_reps, f)
            dpt_reps = dict()
            for dpt_num in tqdm(dpt_dict.keys(), desc="Building rep of DPT for agg " + agg_name + "..."):
                ppl_ft_emb = [ppl_dict[i]["ft"] for i in ppl_dict if ppl_dict[i]["dpt_label"] == dpt_num]
                ppl_skills = [ppl_dict[i]["sk"] for i in ppl_dict if ppl_dict[i]["dpt_label"] == dpt_num]
                if dpt_num not in dpt_reps.keys():
                    dpt_reps[dpt_num] = dict()
                dpt_reps[dpt_num]["ft"] = agg_func(torch.stack(ppl_ft_emb), dim=0)
                dpt_reps[dpt_num]["sk"] = agg_func(torch.stack(ppl_skills), dim=0)
            with open(os.path.join(dd, "dpt_rep_" + agg_name + ".pkl"), "wb") as f:
                pkl.dump(dpt_reps, f)


if __name__ == "__main__":
    global cfg
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    main()
