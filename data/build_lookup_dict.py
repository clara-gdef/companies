import yaml
import os
import pickle as pkl
from tqdm import tqdm


def main():
    dd = cfg["datadir"]
    lookup_ppl = dict()
    all_companies = set()
    all_dpt = set()
    all_clusters = set()
    for split in ["_TEST", "_VALID", "_TRAIN"]:
        print("Loading people...")
        with open(os.path.join(dd, cfg["rep"]['ft']['total'] + split + '.pkl'), 'rb') as f:
            ppl_ft = pkl.load(f)
        with open(os.path.join(dd, cfg["rep"]['sk']['total'] + split + '.pkl'), 'rb') as f:
            ppl_sk = pkl.load(f)
        print("People loaded.")
        for cie in tqdm(ppl_ft.keys(), desc='Builing people lookup for ' + split + '...'):
            all_companies.add(cie)
            for i in ppl_ft[cie].keys():
                all_clusters.add(i)
                if len(ppl_ft[cie][i].keys()) > 0:
                    all_dpt.add((cie, i))
                    for pos, person_id in enumerate(ppl_ft[cie][i]["id_ppl"]):
                        lookup_ppl[person_id] = {"ft": ppl_ft[cie][i]["ppl_emb"][pos],
                                                 "sk": ppl_sk[cie][i]["bin_skills"][pos],
                                                 "cie_name": cie,
                                                 "clus_name": i}
    cie_dict = dict()
    for pos, name in enumerate(sorted(list(all_companies))):
        cie_dict[pos] = name
    offset = pos
    clus_dict = dict()
    for pos, name in enumerate(sorted(list(all_clusters))):
        rank = offset + pos
        clus_dict[rank] = name
    offset = rank
    dpt_dict = dict()
    for pos, name in enumerate(sorted(list(all_dpt))):
        dpt_dict[offset + pos] = name

    print("Num cie: " + str(len(cie_dict)))
    with open(os.path.join(dd, 'lookup_cie.pkl'), 'wb') as f:
        pkl.dump(cie_dict, f)

    print("Num clus: " + str(len(clus_dict)))
    with open(os.path.join(dd, 'lookup_clus.pkl'), 'wb') as f:
        pkl.dump(clus_dict, f)

    print("Num dpt: " + str(len(dpt_dict)))
    with open(os.path.join(dd, 'lookup_dpt.pkl'), 'wb') as f:
        pkl.dump(dpt_dict, f)

    rev_cie_dict = {v: k for k, v in cie_dict.items()}
    rev_clus_dict = {v: k for k, v in clus_dict.items()}
    rev_dpt_dict = {v: k for k, v in dpt_dict.items()}

    print("Num ppl: " + str(len(lookup_ppl)))
    for person in tqdm(lookup_ppl, desc="Adding labels to people..."):
        lookup_ppl[person]["cie_label"] = rev_cie_dict[lookup_ppl[person]["cie_name"]]
        lookup_ppl[person]["clus_labl"] = rev_clus_dict[lookup_ppl[person]["clus_name"]]
        lookup_ppl[person]["dpt_label"] = rev_dpt_dict[(lookup_ppl[person]["cie_name"], lookup_ppl[person]["clus_name"])]

    with open(os.path.join(dd, 'lookup_ppl.pkl'), 'wb') as f:
        pkl.dump(lookup_ppl, f)


if __name__ == "__main__":
    global cfg
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    main()
