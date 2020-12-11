import argparse

import yaml
import os
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
import ipdb
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from tqdm import tqdm


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    high_level = (args.high_level_classes == "True")
    with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
        class_dict = pkl.load(f)
    if not high_level:
        mapper_dict = None
    else:
        mapper_dict = pkl.load(open(os.path.join(CFG["gpudatadir"], "cora_track_to_hl_classes_map.pkl"), 'rb'))

    rev_class_dict = {v: k for k, v in class_dict.items()}

    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data_train = pkl.load(f)

    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "TEST.pkl")
    with open(paper_file, 'rb') as f:
        data_test = pkl.load(f)

    main(args, data_train, data_test, rev_class_dict, mapper_dict, class_dict, high_level)

def main(args, data_train, data_test, rev_class_dict, mapper_dict, class_dict, high_level, class_weights):
    with ipdb.launch_ipdb_on_exception():
        # TRAIN
        cleaned_abstracts, labels = pre_proc_data(data_train, rev_class_dict, mapper_dict)
        train_features = fit_vectorizer(args, cleaned_abstracts)
        if args.model == "SVM":
            model = train_svm(train_features, labels, class_weights)
        elif args.model == "NB":
            model = train_nb(train_features, labels, class_weights)
        else:
            raise Exception("Wrong model type specified, can be either SVM or NB")
        # TEST
        cleaned_abstracts_test, labels_test = pre_proc_data(data_test, rev_class_dict, mapper_dict)
        test_features = fit_vectorizer(args, cleaned_abstracts_test)

        num_c = 10 if high_level else len(class_dict)

        preds_test, preds_test_at_3 = get_predictions(args, model, test_features, labels_test)
        res_at_1_test = eval_model(labels_test, preds_test, num_c, "TEST_" + args.model + "")
        res_at_3_test = eval_model(labels_test, preds_test_at_3, num_c, "TEST_" + args.model + "_@3")

        preds_train, preds_train_at_3 = get_predictions(args, model, train_features, labels)
        res_at_1_train = eval_model(labels, preds_train, num_c, "TRAIN_" + args.model + "")
        res_at_3_train = eval_model(labels, preds_train_at_3, num_c, "TRAIN_" + args.model + "_@3")

        print({**res_at_1_test, **res_at_3_test, **res_at_1_train, **res_at_3_train})
        return {**res_at_1_test, **res_at_3_test, **res_at_1_train, **res_at_3_train}


def get_predictions(args, model, features, labels):
    if args.model == "SVM":
        predictions = model.decision_function(features)
    elif args.model == "NB":
        predictions = model.predict_log_proba(features)

    predictions_at_1 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_1.append(get_pred_at_k(sample, lab, 1))

    predictions_at_3 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_3.append(get_pred_at_k(sample, lab, 3))
    return predictions_at_1, predictions_at_3


def pre_proc_data(data, class_dict, mapper_dict):
    stop_words = set(stopwords.words("english"))
    labels = []
    abstracts = []
    for article in tqdm(data, desc="Parsing articles..."):
        if "Abstract" in article[1].keys() and "class" in article[1].keys():
            if mapper_dict is not None:
                labels.append(mapper_dict[class_dict[article[1]["class"]]])
            else:
                labels.append(class_dict[article[1]["class"]])
            cleaned_ab = [w for w in article[1]["Abstract"] if w not in stop_words]
            abstracts.append(" ".join(cleaned_ab))
    return abstracts, labels


def fit_vectorizer(args, input_data):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_df=args.max_df, min_df=args.min_df)
    ipdb.set_trace()
    print("Fitting vectorizer...")
    data_features = vectorizer.fit_transform(input_data)
    print("Vectorizer fitted.")
    data_features = data_features.toarray()
    return data_features

def train_svm(data, labels, class_weights):
    model = LinearSVC(class_weight=class_weights)
    print("Fitting SVM...")
    model.fit(data, labels)
    print("SVM fitted!")
    return model


def train_nb(data,labels, class_weights):
    priors = [i[1] for i in sorted(class_weights.items())]
    # model = GaussianNB(priors=priors)
    model = MultinomialNB(class_prior=priors)
    print("Fitting Naive Bayes...")
    model.fit(data, labels)
    print("Naive Bayes fitted!")
    return model


def eval_model(labels, preds, num_classes, handle):
    num_c = range(num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_pred_at_k(pred, label, k):
    ranked = np.argsort(pred, axis=-1)
    largest_indices = ranked[::-1][:len(pred)]
    new_pred = largest_indices[0]
    if label in largest_indices[:k]:
        new_pred = label
    return new_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_level_classes", type=str, default="True")
    parser.add_argument("--min_df", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="SVM")
    parser.add_argument("--max_df", type=float, default=0.9)
    parser.add_argument("--class_weight", default=None)
    parser.add_argument("--max_voc_size", type=int, default=40000)
    args = parser.parse_args()
    init(args)
