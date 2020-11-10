import os
import argparse
import pickle as pkl
import yaml
import numpy as np
from tqdm import tqdm


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    high_level = (args.high_level_classes == "True")
    if high_level:
        with open(os.path.join(CFG["gpudatadir"], "cora_high_level_classes_dict.pkl"), 'rb') as f:
            classes = pkl.load(f)
        with open(os.path.join(CFG["gpudatadir"], "cora_track_to_hl_classes_map.pkl"), 'rb') as f:
            high_level_mapper = pkl.load(f)
    else:
        with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
            classes = pkl.load(f)
        high_level_mapper = None
    tgt_file = os.path.join(CFG["gpudatadir"], "cora_embedded_" + args.ft_type + "_TRAIN.pkl")
    with open(tgt_file, 'rb') as f:
        train_dataset = pkl.load(f)

    classes_rep = []
    for class_num in tqdm(classes.keys()):
        papers = select_relevant_papers(train_dataset, class_num, high_level_mapper)
        classes_rep.append(np.mean(np.stack(papers), axis=0))

    assert len(classes_rep) == len(classes)

    if high_level:
        tgt_file = os.path.join(CFG["gpudatadir"], "areas_reps_" + args.ft_type + ".pkl")
    else:
        tgt_file = os.path.join(CFG["gpudatadir"], "tracks_reps_" + args.ft_type + ".pkl")
    with open(tgt_file, 'wb') as f:
        pkl.dump(classes_rep, f)


def select_relevant_papers(train_dataset, class_num, high_level_mapper):
    selected_papers = []
    for paper in train_dataset:
        if high_level_mapper is not None:
            if high_level_mapper[paper["class"]] == class_num:
                selected_papers.append(paper["avg_profile"])
        else:
            if paper["class"] == class_num:
                selected_papers.append(paper["avg_profile"])
    assert len(selected_papers) > 0
    return selected_papers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default="ft_en.bin")
    parser.add_argument("--high_level_classes", type=str, default="True")
    args = parser.parse_args()
    main(args)
