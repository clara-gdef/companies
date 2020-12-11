import argparse
import os
import pickle as pkl
import ipdb
import torch
import yaml
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    data = []
    for item in ["TRAIN", "VALID", "TEST"]:
        paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["emb"] + "_fs_" + item + ".pkl")
        with open(paper_file, 'rb') as f:
            data.extend(pkl.load(f))
    classes = Counter()

    high_level = (args.high_level_classes == "True")

    mapper_dict = pkl.load(open(os.path.join(CFG["gpudatadir"], "cora_track_to_hl_classes_map.pkl"), 'rb'))
    for paper in tqdm(data):
        if high_level:
            classes[mapper_dict[paper["class"]]] += 1
        else:
            classes[paper["class"]] += 1
    mc_class = classes.most_common(1)[0][0]
    fqc = {k : v/ sum(classes.values()) for k, v in classes.items()}
    fqc_sorted_tsr = torch.FloatTensor([fqc[i] for i in sorted(fqc.keys())])
    ipdb.set_trace()
    with open(os.path.join(CFG["gpudatadir"], "cora_area_fqc.pkl"), 'wb') as f:
        pkl.dump(fqc_sorted_tsr, f)

    ipdb.set_trace()
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["emb"] + "_fs_TEST.pkl")
    with open(paper_file, 'rb') as f:
        data_test = pkl.load(f)

    labels = []
    for paper in tqdm(data_test):
        if high_level:
            labels.append(mapper_dict[paper["class"]])
        else:
            labels.append(paper["class"])
    preds = [mc_class] * len(labels)

    with ipdb.launch_ipdb_on_exception():
        res_1 = get_metrics(preds, labels, len(classes), "mc_@1", 0)
        print(res_1)
        mc_class_10 = [[i[0] for i in classes.most_common(10)]] * len(labels)
        res_2 = get_metrics_at_k(mc_class_10, labels, len(classes), "mc_@10", 0)
        print(res_2)

        ipdb.set_trace()


def get_metrics(preds, labels, num_classes, handle, offset):
    num_c = range(offset, offset + num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_metrics_at_k(predictions, labels, num_classes, handle, offset):
    out_predictions = []
    transformed_predictions = predictions + offset
    for index, pred in enumerate(transformed_predictions):
        if labels[index] in pred:
            out_predictions.append(labels[index])
        else:
            if type(pred[0]) == torch.Tensor:
                out_predictions.append(pred[0].item())
            else:
                out_predictions.append(pred[0])
    return get_metrics(out_predictions, labels, num_classes, handle, offset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_level_classes", type=str, default="True")
    args = parser.parse_args()
    main(args)
