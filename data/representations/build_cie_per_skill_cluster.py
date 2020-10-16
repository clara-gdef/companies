import argparse
import json
import math
import operator
import yaml
from collections import Counter

import torch
import os
import pickle as pkl
from tqdm import tqdm
import ipdb
import numpy as np
from sklearn.cluster import KMeans


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    with ipdb.launch_ipdb_on_exception():
        num_c = args.num_clusters
        year_start = args.year_start
        year_end = args.year_end

        with open(os.path.join(CFG["datadir"], args.skill_file), "rb") as f:
            skills_classes = pkl.load(f)

        skill_dict = {k: v for k, v in enumerate(skills_classes)}
        rev_skill_dict = {v: k for k, v in enumerate(skills_classes)}

        for split in ["TRAIN", "VALID", "TEST"]:
            if args.temporal:
                target = os.path.join(CFG["datadir"], args.ft_type + "_partial_rep_1990_2018_" + split + ".pkl")
                with open(target, "rb") as f:
                    cie_dict = pkl.load(f)

                clusters_dict = dict()
                cie_names = get_all_cie()

                for year in tqdm(range(year_start, year_end), desc="Building " + split + " dataset..."):
                    year_profiles_emb = []
                    year_profiles_id = []
                    corresponding_cie = dict()

                    for i, c in enumerate(cie_dict[year].keys()):
                        year_profiles_emb.extend(cie_dict[year][c]["profiles"])
                        year_profiles_id.extend(cie_dict[year][c]["id"])
                        for j in cie_dict[year][c]["id"]:
                            corresponding_cie[j] = c

                    mapping_id_position = {identifier: position for position, identifier in enumerate(year_profiles_id)}
                    rev_mapping_id_position = {v: k for k, v in mapping_id_position.items()}

                    if len(year_profiles_id) > args.min_ppl_count:

                        clusters_dict[year] = {}
                        skill_predictions, clusters = compute_clusters(num_c, year_profiles_emb, classifier)
                        tfidf_per_clusters = compute_tfidf(num_c, skill_dict, rev_skill_dict, skill_predictions)
                        for cie in cie_names:
                            clusters_dict[year][cie] = {}
                            for id_c in range(num_c):
                                clusters_dict[year][cie][id_c] = {}
                                ordered_skills = sorted(tfidf_per_clusters[id_c].items(), key=operator.itemgetter(1), reverse=True)
                                selected_ppl_id = select_ppl_id(year_profiles_id, clusters, id_c, corresponding_cie, cie)
                                selected_ppl_emb = select_ppl_emb(year_profiles_emb, clusters, id_c, rev_mapping_id_position, selected_ppl_id)
                                # not all cie are in all clusters
                                if len(selected_ppl_id) > 0:
                                    clusters_dict[year][cie][id_c]["skills_list"] = ordered_skills
                                    clusters_dict[year][cie][id_c]["id_ppl"] = selected_ppl_id
                                    clusters_dict[year][cie][id_c]["ppl_emb"] = torch.FloatTensor(selected_ppl_emb)
                                    clusters_dict[year][cie][id_c]["cie"] = [corresponding_cie[i] for i in selected_ppl_id]
                                    assert len(selected_ppl_id) == len(selected_ppl_emb)
                print("Saving dataset...")
                fname = "reps/ft_emb/" + args.ft_type + "_" + str(args.num_clusters) + "_" + args.tgt_file + "_" + split + ".pkl"
                file_path = os.path.join(CFG["datadir"], fname)
                print("location: " + file_path)
                with open(file_path, "wb") as f:
                    pkl.dump(clusters_dict, f)
            else:
                target = os.path.join(CFG["datadir"], "total_rep_" + split + "_standardized.pkl")
                with open(target, "rb") as f:
                    cie_dict = pkl.load(f)

                clusters_dict = dict()
                cie_names = get_all_cie()

                profiles_emb = []
                profiles_id = []
                corresponding_cie = dict()
                for i, c in enumerate(cie_dict.keys()):
                    profiles_emb.extend(cie_dict[c]["profiles"])
                    profiles_id.extend(cie_dict[c]["id"])
                    for j in cie_dict[c]["id"]:
                        corresponding_cie[j] = c

                mapping_id_position = {identifier: position for position, identifier in enumerate(profiles_id)}
                rev_mapping_id_position = {v: k for k, v in mapping_id_position.items()}

                clusters = compute_clusters(num_c, profiles_emb)

                for cie in tqdm(cie_names, desc="processing dataset..."):
                    clusters_dict[cie] = {}
                    for id_c in range(num_c):
                        clusters_dict[cie][id_c] = {}
                        selected_ppl_id = select_ppl_id(profiles_id, clusters, id_c, corresponding_cie,
                                                        cie)
                        selected_ppl_emb = select_ppl_emb(profiles_emb, clusters, id_c,
                                                          rev_mapping_id_position, selected_ppl_id)
                        if len(selected_ppl_id) > 0:
                            clusters_dict[cie][id_c]["id_ppl"] = selected_ppl_id
                            clusters_dict[cie][id_c]["ppl_emb"] = torch.FloatTensor(selected_ppl_emb)
                            clusters_dict[cie][id_c]["cie"] = [corresponding_cie[i] for i in selected_ppl_id]
                            assert len(selected_ppl_id) == len(selected_ppl_emb)
                print("Saving dataset...")
                fname = "reps/ft_emb/" + args.ft_type + "_" + str(
                    args.num_clusters) + "_" + args.tgt_file + "_" + split + ".pkl"
                file_path = os.path.join(CFG["datadir"], fname)
                print("location: " + file_path)
                with open(file_path, "wb") as f:
                    pkl.dump(clusters_dict, f)


def compute_tfidf(doc_number, skill_dict, rev_skill_dict, predictions):
    flat_skills = set([j for i in zip(skill_dict.values()) for j in i])
    # term frequency per document.
    # Here, it can only appear once in each document so we simplify the computations
    tf = dict()
    for skill in flat_skills:
        tf[skill] = dict()
        for ten, tensor in enumerate(predictions):
            if tensor[rev_skill_dict[skill]] == 1:
                # determines the frequency of the skill amongst the predicted skills
                tf[skill][ten] = 1

    idf = dict()
    for skill in flat_skills:
        if len(tf[skill]) > 0:
            idf[skill] = math.log(doc_number / len(tf[skill]))

    tfidf_per_skill = dict()
    for skill in flat_skills:
        if skill in tf.keys() and skill in idf.keys():
            tfidf_per_skill[skill] = dict()
            for ten, tensor in enumerate(predictions):
                if ten in tf[skill].keys():
                    tfidf_per_skill[skill][ten] = tf[skill][ten] * idf[skill]
                else:
                    tfidf_per_skill[skill][ten] = 0.

    tfidf_per_clus = {k: dict() for k in range(doc_number)}
    for sk in tfidf_per_skill.keys():
        for clus in range(doc_number):
            if tfidf_per_skill[sk][clus] > 0.0:
                tfidf_per_clus[clus][sk] = tfidf_per_skill[sk][clus]

    return tfidf_per_clus


def select_ppl_id(year_profiles_id, clusters, id_c, corresponding_cie, cie):
    selected = []
    for j, i in enumerate(year_profiles_id):
        if clusters.labels_[j] == id_c:
            if corresponding_cie[i] == cie:
                selected.append(i)
    selected_set = set(selected)
    selected = [i for i in selected_set]
    return selected


def select_ppl_emb(profiles_emb, clusters, id_c, rev_mapping_id_position, selected_ppl_id):
    selected = []
    for j, i in enumerate(profiles_emb):
        if clusters.labels_[j] == id_c:
            if j in rev_mapping_id_position.keys():
                if rev_mapping_id_position[j] in selected_ppl_id:
                    selected.append(i)
    return selected


def compute_clusters(num_c, profiles):
    centers = "k-means++"
    print("Computing " + str(args.num_clusters) + " clusters...")
    clusters = KMeans(n_clusters=num_c, init=centers).fit(np.asarray(profiles))
    return clusters


def build_skills_dicts(predictions, skill_dict):
    tmp = dict()
    counter = Counter()
    for ten, tensor in enumerate(predictions):
        tmp[ten] = []
        for ind, skill in enumerate(tensor):
            if skill.item() == 1:
                counter[ind] += 1
                tmp[ten].append(skill_dict[ind])

    sk = dict()
    for ten, tensor in enumerate(predictions):
        sk[ten] = []
        min_count = len(predictions)
        min_count_ind = 0
        for ind, skill in enumerate(tensor):
            if skill.item() == 1 and counter[ind] < min_count:
                min_count = counter[ind]
                min_count_ind = ind
        sk[ten].append(skill_dict[min_count_ind])

    return sk


def get_all_cie():
    cie_file = os.path.join(CFG["datadir"], "cie_list.pkl")
    with open(cie_file, 'rb') as f:
        final_cie = pkl.load(f)
    return final_cie


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_file", type=str, default="good_skills.p")
    parser.add_argument("--model_dir", type=str, default='/net/big/gainondefor/work/trained_models/companies')
    parser.add_argument("--num_clusters", type=int, default=30)
    parser.add_argument("--max_skill_number", type=int, default=5)
    parser.add_argument("--tgt_file", type=str, default="latest_clus_cie_per_skills")
    parser.add_argument("--ft_type", type=str, default="fs")
    parser.add_argument("--temporal", type=bool, default=False)
    parser.add_argument("--min_ppl_count", type=int, default=200)
    parser.add_argument("--year_start", type=int, default=1996)
    parser.add_argument("--year_end", type=int, default=2017)
    args = parser.parse_args()
    main(args)
