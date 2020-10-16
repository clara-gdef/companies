import os
import argparse
import pickle as pkl
from collections import Counter

import ipdb
import torch
import yaml
from tqdm import tqdm


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        skill_file = os.path.join(CFG["datadir"], args.skill_index)
        with open(skill_file, 'rb') as f:
            skills = pkl.load(f)
        skills_dict = {name: ind for ind, name in enumerate(sorted(skills))}

        for split in ["VALID", "TEST", "TRAIN"]:
            clustered_cie,  skilled_profiles = init(args, split)

            skilled_profiles_dict = {}
            for person in tqdm(skilled_profiles, desc='building profile dict for ' + split + ' set...'):
                skilled_profiles_dict[person[0]] = turn_skills_to_vector(person[1], skills_dict)

            final_dict = {}
            if args.temporal:
                for year in tqdm(clustered_cie.keys(), desc='building full dataset for ' + split + ' set...'):
                    final_dict[year] = {}
                    for cie_name in clustered_cie[year].keys():
                        final_dict[year][cie_name] = {}
                        for clus_number in clustered_cie[year][cie_name].keys():
                            final_dict[year][cie_name][clus_number] = {"id_ppl": [], "bin_skills": []}
                            # not all cie are in all clusters
                            if len(clustered_cie[year][cie_name][clus_number].keys()) > 0:
                                for person_id in clustered_cie[year][cie_name][clus_number]["id_ppl"]:
                                    if person_id in skilled_profiles_dict.keys():
                                        final_dict[year][cie_name][clus_number]["id_ppl"].append(person_id)
                                        final_dict[year][cie_name][clus_number]["bin_skills"].append(skilled_profiles_dict[person_id])

                f_name = 'reps/skills/' + str(args.num_clusters) + "_clus_" + args.output_file + "_" + split + "_standardized.pkl"
                final_file = os.path.join(CFG["datadir"], f_name)
                with open(final_file, "wb") as f:
                    pkl.dump(final_dict, f)
            else:
                for cie_name in tqdm(clustered_cie.keys()):
                    final_dict[cie_name] = {}
                    for clus_number in clustered_cie[cie_name].keys():
                        final_dict[cie_name][clus_number] = {"id_ppl": [], "bin_skills": []}
                        if len(clustered_cie[cie_name][clus_number].keys()) > 0:
                            for person_id in clustered_cie[cie_name][clus_number]["id_ppl"]:
                                if person_id in skilled_profiles_dict.keys():
                                    final_dict[cie_name][clus_number]["id_ppl"].append(person_id)
                                    final_dict[cie_name][clus_number]["bin_skills"].append(
                                        skilled_profiles_dict[person_id])
                print("Constructed companies: " + str(len(final_dict)))
                total_ppl_per_cie = Counter()
                for cie_name in clustered_cie.keys():
                    for clus_number in clustered_cie[cie_name].keys():
                        total_ppl_per_cie[cie_name] += len(clustered_cie[cie_name][clus_number])
                for cie_name in clustered_cie.keys():
                    if total_ppl_per_cie[cie_name] == 0:
                        final_dict.pop(cie_name)
                print("Remaning companies: " + str(len(final_dict)))
                f_name = 'reps/skills/' + str(args.num_clusters) + "_clus_" + args.output_file + "_" + split + "_standardized.pkl"
                final_file = os.path.join(CFG["datadir"], f_name)
                with open(final_file, "wb") as f:
                    pkl.dump(final_dict, f)


def init(args, split):
    num_c = args.num_clusters
    fname = "reps/ft_emb/fs_" + str(num_c) + "_" + args.clusters_cie + "_" + split + "_standardized.pkl"
    clusters_file = os.path.join(CFG["datadir"], fname)
    with open(clusters_file, 'rb') as f:
        clustered_cie = pkl.load(f)

    profile_file = os.path.join(CFG["datadir"], args.profiles_w_skills + "_" + split + ".pkl")
    with open(profile_file, 'rb') as f:
        skilled_profiles = pkl.load(f)

    return clustered_cie, skilled_profiles


def turn_skills_to_vector(skill_list, skill_dict):
    result = torch.zeros(1, len(skill_dict))
    for skill in skill_list:
        if skill in skill_dict.keys():
            result[0, skill_dict[skill]] = 1.
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clusters", type=int, default=30)
    parser.add_argument("--profiles_w_skills", type=str, default='profiles_jobs_skills')
    parser.add_argument("--clusters_cie", type=str, default='latest_clus_cie_per_skills')
    parser.add_argument("--skill_index", type=str, default='good_skills.p')
    parser.add_argument("--temporal", type=bool, default=False)
    parser.add_argument("--output_file", type=str, default='latest_ppl_as_bin_skills')
    args = parser.parse_args()
    main(args)
