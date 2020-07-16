import yaml
import os
import pickle as pkl
from tqdm import tqdm
import torch

def main():
    dd = cfg["datadir"]
    with open(os.path.join(dd, 'lookup_ppl.pkl'), 'rb') as f:
        ppl_dict = pkl.load(f)
    with open(os.path.join(dd, 'lookup_cie.pkl'), 'rb') as f:
        cie_dict = pkl.load(f)
    with open(os.path.join(dd, 'lookup_clus.pkl'), 'rb') as f:
        clus_dict = pkl.load(f)
    with open(os.path.join(dd, 'lookup_dpt.pkl'), 'rb') as f:
        dpt_dict = pkl.load(f)

    print("Loading people...")
    with open(os.path.join(dd, cfg["rep"]['ft']['total'] + '_TRAIN.pkl'), 'rb') as f:
        ppl_ft = pkl.load(f)
    with open(os.path.join(dd, cfg["rep"]['sk']['total'] + '_TRAIN.pkl'), 'rb') as f:
        ppl_sk = pkl.load(f)
    print("People loaded.")

    cie_reps = dict()
    clus_reps = dict()
    dpt_reps = dict()
    for (agg_name, agg_func) in [("avg", torch.mean), ("max", torch.max), ("sum", torch.sum)]:
        for cie_num, cie_name in cie_dict.items():
            ppl_in_cie = [ppl_ft[cie_name][id_clus]["id_ppl"] for id_clus in ppl_ft[cie_name].keys() if len(ppl_ft[cie_name][id_clus].keys) > 0]
            ppl_ft_emb = [ppl_dict[i]["ft"] for i in ppl_in_cie]
            ppl_skills = [ppl_dict[i]["sk"] for i in ppl_in_cie]
            cie_reps[cie_num]["ft"] = agg_func(torch.stack(ppl_ft_emb))
            cie_reps[cie_num]["sk"] = agg_func(torch.stack(ppl_skills))

if __name__ == "__main__":
    global cfg
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    main()
