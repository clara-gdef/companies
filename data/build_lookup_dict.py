import os
import pickle as pkl
import yaml
import ipdb
from tqdm import tqdm


def main():
    with ipdb.launch_ipdb_on_exception():

        dat_dir = CFG["datadir"]
        lookup_ppl, all_companies, all_clusters, all_dpt = build_ppl_bag_sets(dat_dir)
        cie_dict, clus_dict, dpt_dict = build_bag_dict(dat_dir, all_companies, all_clusters, all_dpt)

        rev_cie_dict = {v: k for k, v in cie_dict.items()}
        rev_clus_dict = {v: k for k, v in clus_dict.items()}
        rev_dpt_dict = {v: k for k, v in dpt_dict.items()}

        num_cie = len(cie_dict)
        num_clus = len(clus_dict)
        num_dpt = len(dpt_dict)

        print("Num ppl: " + str(len(lookup_ppl)))
        for person in tqdm(lookup_ppl, desc="Adding labels to people..."):
            cie_lab = rev_cie_dict[lookup_ppl[person]["cie_name"]]
            assert 0 <= cie_lab <= num_cie - 1
            lookup_ppl[person]["cie_label"] = cie_lab

            clus_lab = rev_clus_dict[lookup_ppl[person]["clus_name"]]
            assert num_cie <= clus_lab <= num_cie + num_clus - 1
            lookup_ppl[person]["clus_label"] = clus_lab

            lookup_ppl[person]["dpt_name"] = (lookup_ppl[person]["cie_name"],
                                              lookup_ppl[person]["clus_name"])

            dpt_lab = rev_dpt_dict[lookup_ppl[person]["dpt_name"]]
            assert num_cie + num_clus <= dpt_lab <= num_cie + num_clus + num_dpt - 1
            lookup_ppl[person]["dpt_label"] = dpt_lab

        with open(os.path.join(dat_dir, 'lookup_ppl.pkl'), 'wb') as f_name:
            pkl.dump(lookup_ppl, f_name)


def build_ppl_bag_sets(dat_dir):
    lookup_ppl = dict()
    all_companies = set()
    all_dpt = set()
    all_clusters = set()
    for split in ["_TEST", "_VALID", "_TRAIN"]:
        print("Loading people...")
        with open(os.path.join(dat_dir, CFG["rep"]['ft']['total'] + split + '.pkl'), 'rb') as f_name:
            ppl_ft = pkl.load(f_name)
        # with open(os.path.join(dat_dir, CFG["rep"]['sk']['total'] + split + '.pkl'), 'rb') as f_name:
        #     ppl_sk = pkl.load(f_name)
        print("People loaded.")
        for cie in tqdm(ppl_ft.keys(), desc='Builing people lookup for ' + split + '...'):
            all_companies.add(cie)
            for i in ppl_ft[cie].keys():
                all_clusters.add(i)
                if len(ppl_ft[cie][i].keys()) > 0:
                    all_dpt.add((cie, i))
                    for pos, person_id in enumerate(ppl_ft[cie][i]["id_ppl"]):
                        lookup_ppl[person_id] = {"ft": ppl_ft[cie][i]["ppl_emb"][pos],
                                                 # "sk": ppl_sk[cie][i]["bin_skills"][pos],
                                                 "cie_name": cie,
                                                 "clus_name": i}
    return lookup_ppl, all_companies, all_clusters, all_dpt


def build_bag_dict(dat_dir, all_companies, all_clusters, all_dpt):
    cie_dict = dict()
    for pos, name in enumerate(sorted(list(all_companies))):
        cie_dict[pos] = name
    offset = pos + 1
    clus_dict = dict()
    for pos, name in enumerate(sorted(list(all_clusters))):
        rank = offset + pos
        clus_dict[rank] = name
    offset = rank + 1
    dpt_dict = dict()
    for pos, name in enumerate(sorted(list(all_dpt))):
        dpt_dict[offset + pos] = name

    print("Num cie: " + str(len(cie_dict)))
    with open(os.path.join(dat_dir, 'lookup_cie.pkl'), 'wb') as f_name:
        pkl.dump(cie_dict, f_name)

    print("Num clus: " + str(len(clus_dict)))
    with open(os.path.join(dat_dir, 'lookup_clus.pkl'), 'wb') as f_name:
        pkl.dump(clus_dict, f_name)

    print("Num dpt: " + str(len(dpt_dict)))
    with open(os.path.join(dat_dir, 'lookup_dpt.pkl'), 'wb') as f_name:
        pkl.dump(dpt_dict, f_name)

    return cie_dict, clus_dict, dpt_dict


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main()
