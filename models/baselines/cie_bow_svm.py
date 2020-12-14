import argparse

import yaml
import os
import torch
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import ipdb
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from tqdm import tqdm


def init(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with open(os.path.join(CFG["gpudatadir"], "lookup_cie.pkl"), 'rb') as f:
        class_dict = pkl.load(f)
    rev_class_dict = {v: k for k, v in class_dict.items()}

    data_train, data_test = get_labelled_data(args)

    class_weights = get_class_weights(data_train)

    main(args, data_train, data_test, class_dict, class_weights)


def get_labelled_data(args):
    if args.load_dataset == "True":
        with open(os.path.join(CFG["gpudatadir"], "txt_profiles_cie_labelled_TRAIN.pkl"), 'rb') as f_name:
            data_train = pkl.load(f_name)
        with open(os.path.join(CFG["gpudatadir"], "txt_profiles_cie_labelled_TEST.pkl"), 'rb') as f_name:
            data_test = pkl.load(f_name)
    else:
        paper_file = os.path.join(CFG["gpudatadir"], "profiles_jobs_skills_TRAIN.pkl")
        with open(paper_file, 'rb') as f:
            raw_profiles_train = pkl.load(f)

        file_name = "disc_poly_avg_ft_TRAIN_standardized.pkl"
        with open(os.path.join(CFG["gpudatadir"], file_name), 'rb') as f_name:
            labelled_train = torch.load(f_name)

        data_train = map_profiles_to_label(raw_profiles_train, labelled_train["tuples"])
        with open(os.path.join(CFG["gpudatadir"], "txt_profiles_cie_labelled_TRAIN.pkl"), 'wb') as f_name:
            pkl.dump(data_train, f_name)

        paper_file = os.path.join(CFG["gpudatadir"], "profiles_jobs_skills_TEST.pkl")
        with open(paper_file, 'rb') as f:
            raw_profiles_test = pkl.load(f)

        file_name = "disc_poly_avg_ft_TEST_standardized.pkl"
        with open(os.path.join(CFG["gpudatadir"], file_name), 'rb') as f_name:
            labelled_test = torch.load(f_name)

        data_test = map_profiles_to_label(raw_profiles_test, labelled_test["tuples"])
        with open(os.path.join(CFG["gpudatadir"], "txt_profiles_cie_labelled_TEST.pkl"), 'wb') as f_name:
            pkl.dump(data_test, f_name)

    return data_train, data_test



def main(args, data_train, data_test, class_dict, class_weights):
    with ipdb.launch_ipdb_on_exception():
        # TRAIN
        cleaned_profiles, labels = pre_proc_data(data_train)
        train_features, vectorizer = fit_vectorizer(args, cleaned_profiles)
        if args.model == "SVM":
            model = train_svm(train_features, labels, class_weights)
        elif args.model == "NB":
            model = train_nb(train_features, labels, class_weights)
        else:
            raise Exception("Wrong model type specified, can be either SVM or NB")
        # TEST
        cleaned_abstracts_test, labels_test = pre_proc_data(data_test)
        test_features = vectorizer.transform(cleaned_abstracts_test)

        num_c = len(class_dict)

        preds_test, preds_test_at_10 = get_predictions(args, model, test_features, labels_test)
        res_at_1_test = eval_model(labels_test, preds_test, num_c, "TEST_" + args.model + "")
        res_at_10_test = eval_model(labels_test, preds_test_at_10, num_c, "TEST_" + args.model + "_@10")

        preds_train, preds_train_at_10 = get_predictions(args, model, train_features, labels)
        res_at_1_train = eval_model(labels, preds_train, num_c, "TRAIN_" + args.model + "")
        res_at_10_train = eval_model(labels, preds_train_at_10, num_c, "TRAIN_" + args.model + "_@10")

        print({**res_at_1_test, **res_at_10_test, **res_at_1_train, **res_at_10_train})
        return {**res_at_1_test, **res_at_10_test, **res_at_1_train, **res_at_10_train}


def map_profiles_to_label(raw_profiles, labelled_data):
    mapped_profiles = {}
    for person in tqdm(raw_profiles, desc='parsing raw profiles...'):
        person_id = person[0]
        for item in labelled_data:
            if item["id"] == person_id:
                mapped_profiles[person_id] = {}
                mapped_profiles[person_id]["jobs"] = handle_jobs(person[3])
                mapped_profiles[person_id]["cie"] = item["cie"]
    return mapped_profiles

def handle_jobs(job_list):
    exp_list = []
    for job in sorted(job_list, key=lambda k: k['to'], reverse=True):
        for sentence in job["job"]:
            exp_list.append(sentence)
    tmp = ' '.join([i for i in exp_list])
    return tmp


def get_predictions(args, model, features, labels):
    if args.model == "SVM":
        predictions = model.decision_function(features)
    elif args.model == "NB":
        predictions = model.predict_log_proba(features)

    predictions_at_1 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_1.append(get_pred_at_k(sample, lab, 1))

    predictions_at_10 = []
    for sample, lab in zip(predictions, labels):
        predictions_at_10.append(get_pred_at_k(sample, lab, 10))
    return predictions_at_1, predictions_at_10


def pre_proc_data(data):
    stop_words = set(stopwords.words("french"))
    labels = []
    profiles = []
    for k in tqdm(data.keys(), desc="Parsing profiles..."):
        labels.append(data[k]["cie"])
        cleaned_ab = [w for w in data[k]["jobs"].split(" ") if (w not in stop_words) and (w != "")]
        profiles.append(" ".join(cleaned_ab))
    return profiles, labels


def fit_vectorizer(args, input_data):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_df=args.max_df, min_df=args.min_df)
    print("Fitting vectorizer...")
    data_features = vectorizer.fit_transform(input_data)
    print("Vectorizer fitted.")
    data_features = data_features.toarray()
    return data_features, vectorizer

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


def get_class_weights(data_train):
    class_counter = Counter()
    for k in data_train.keys():
        class_counter[data_train[k]["cie"]] +=1
    total_samples = sum([i for i in class_counter.values()])
    class_weight = {k: v/total_samples for k, v in class_counter.items()}
    return class_weight


def get_pred_at_k(pred, label, k):
    ranked = np.argsort(pred, axis=-1)
    largest_indices = ranked[::-1][:len(pred)]
    new_pred = largest_indices[0]
    if label in largest_indices[:k]:
        new_pred = label
    return new_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dataset", type=str, default="False")
    parser.add_argument("--min_df", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="SVM")
    parser.add_argument("--max_df", type=float, default=0.9)
    parser.add_argument("--class_weight", default=None)
    parser.add_argument("--max_voc_size", type=int, default=40000)
    args = parser.parse_args()
    init(args)
