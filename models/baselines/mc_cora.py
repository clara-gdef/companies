import os
import pickle as pkl
import ipdb
import torch
import yaml
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["emb"] + "_fs_TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data = pkl.load(f)
    classes = Counter()
    for paper in tqdm(data):
        classes[paper["class"]] += 1
    mc_class = classes.most_common(1)[0]
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["emb"] + "_fs_TEST.pkl")
    with open(paper_file, 'rb') as f:
        data_test = pkl.load(f)

    labels = []
    for paper in tqdm(data_test):
        labels.append(paper["class"])
    preds = [mc_class] * len(labels)
    res_1 = get_metrics(preds, labels, 70, "mc_@1", 0)
    print(res_1)

    mc_class_10 = [i[0] for i in classes.most_common(10)]
    res_2 = get_metrics_at_k(mc_class_10, labels, 70, "mc_@1", 0)
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
        if labels[index].item() in pred:
            out_predictions.append(labels[index].item())
        else:
            if type(pred[0]) == torch.Tensor:
                out_predictions.append(pred[0].item())
            else:
                out_predictions.append(pred[0])
    return get_metrics(out_predictions, labels, num_classes, handle, offset)

if __name__ == "__main__":
    main()
