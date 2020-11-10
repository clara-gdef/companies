import yaml
import os
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorize
from sklearn.svm import LinearSVC
import ipdb
from nltk.corpus import stopwords

from tqdm import tqdm


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "_fs_TRAIN.pkl")
    with open(paper_file, 'rb') as f:
        data_train = pkl.load(f)

    cleaned_data = pre_proc_data(data_train)

    paper_file = os.path.join(CFG["gpudatadir"], CFG["rep"]["cora"]["papers"]["plain"] + "_fs_TEST.pkl")
    with open(paper_file, 'rb') as f:
        data_test = pkl.load(f)


def pre_proc_data(data):
    # tokenize
    # remove stop words
    sw = set(stopwords.words("english"))
    # fit vectorizer

    ipdb.set_trace()


def train_svm():
    pass


if __name__ == "__main__":
    main()
