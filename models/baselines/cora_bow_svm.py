import yaml
import os
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import ipdb
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from tqdm import tqdm


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
        class_dict = pkl.load(f)
    rev_class_dict = {v: k for k, v in class_dict.items()}
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data_train = pkl.load(f)

    # paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "VALID.pkl")
    # with open(paper_file, 'rb') as f:
    #     data_valid = pkl.load(f)

    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "TEST.pkl")
    with open(paper_file, 'rb') as f:
        data_test = pkl.load(f)

    with ipdb.launch_ipdb_on_exception():
        # TRAIN
        cleaned_abstracts, labels = pre_proc_data(data_train, rev_class_dict)
        train_features = fit_vectorizer(cleaned_abstracts)
        model = train_svm(train_features, labels)

        # TEST
        cleaned_abstracts_test, labels_test = pre_proc_data(data_test, rev_class_dict)
        test_features = fit_vectorizer(cleaned_abstracts_test)
        predicted_labels = model.predict(test_features)
        print(model.score(test_features, labels_test))
        ipdb.set_trace()

        res = eval_model(labels_test, predicted_labels)

        print(res)


def pre_proc_data(data, class_dict):
    stop_words = set(stopwords.words("english"))
    labels = []
    abstracts = []
    for article in tqdm(data, desc="Parsing articles..."):
        if "Abstract" in article[1].keys() and "class" in article[1].keys():
            labels.append(class_dict[article[1]["class"]])
            cleaned_ab = [w for w in article[1]["Abstract"] if w not in stop_words]
            abstracts.append(" ".join(cleaned_ab))
    return abstracts, labels


def fit_vectorizer(input_data):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=1500)
    print("Fitting vectorizer...")
    data_features = vectorizer.fit_transform(input_data)
    print("Vectorizer fitted.")
    data_features = data_features.toarray()
    return data_features


def train_svm(data, labels):
    model = LinearSVC()
    print("Fitting SVM...")
    model.fit(data, labels)
    print("SVM fitted!")
    return model

def eval_model(labels, predicted_labels):
    num_c = range(offset, offset + num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}


if __name__ == "__main__":
    main()
