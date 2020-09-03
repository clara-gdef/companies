import os
import pickle as pkl
import yaml
import fastText
import re
import argparse
import ipdb
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors


def main(args):
    with ipdb.launch_ipdb_on_exception():
        global CFG
        with open("config.yaml", "r") as ymlfile:
            CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        estimator, ft_model = get_nn_estimator()
        vocab_lookup = get_vocab_lookup(ft_model)


        print("Loading data...")
        with open(os.path.join(CFG["gpudatadir"], "lookup_ppl.pkl"), 'rb') as f_name:
            ppl_lookup = pkl.load(f_name)
        print("Data loaded.")


        key = 708432
        person_rep = ppl_lookup[key]["ft"]

        ipdb.set_trace()


def get_nn_estimator():
    estimator = NearestNeighbors(n_neighbors=10, radius=1.0)
    if not args.load_nn_estimator:
        estimator, ft_model = fit_nearest_neighbors(estimator)
    else:
        with open(os.path.join(CFG["modeldir"], "NN_estimator"), 'rb') as f_name:
            est_params, ft_model = pkl.load(f_name)
        estimator.set_params(est_params)
    return estimator, ft_model


def get_vocab_lookup(ft_model):
    if not args.load_lookup_vocabulary:
        vocab_lookup = build_voc_lookup(ft_model)
    else:
        with open(os.path.join(CFG["gpudatadir"], "ft_vocab_lookup.pkl"), "rb") as f:
            vocab_lookup = pkl.load(f)
    return vocab_lookup


def fit_nearest_neighbors(estimator):
    print("Loading model...")
    ft_model = fastText.load_model("/net/big/gainondefor/work/trained_models/companies/ft_fs.bin")
    input_matrix = ft_model.get_input_matrix()
    print("Model loaded.")
    print("Fitting estimator...")
    estimator.fit(input_matrix)
    print("Estimator learned.")
    with open(os.path.join(CFG["modeldir"], "NN_estimator"), 'wb') as f_name:
        pkl.dump(estimator.get_params(), f_name)
    return estimator, ft_model


def build_voc_lookup(ft_model):
    vocab_lookup = {}
    if not args.load_vocabulary:
        vocab = build_vocab()
    else:
        with open(os.path.join(CFG["gpudatadir"], "ft_vocab.pkl"), "rb") as f:
            vocab = pkl.load(f)
    for ind, word in tqdm(enumerate(vocab), desc="Building vocab lookup..."):
        vocab_lookup[ind] = ft_model.get_word_vector(str(word))

    with open(os.path.join(CFG["gpudatadir"], "ft_vocab_lookup.pkl"), "wb") as f:
        pkl.dump(vocab_lookup, f)
    return vocab_lookup


def build_vocab():
    print("Building vocab...")
    vocab_set = set()
    file_path = os.path.join(CFG["gpudatadir"], "ft_unsupervised_input.txt")
    with open(file_path, 'r') as f:
        for line in f:
            vocab_set.update(re.split(r'[\W]', line))
    vocab_set.remove('')
    voc_list = sorted([i for i in vocab_set])
    vocab = {}
    for ind, word in tqdm(enumerate(voc_list), desc="Finalizing vocab..."):
        vocab[ind] = word
    print("Vocab built.")
    with open(os.path.join(CFG["gpudatadir"], "ft_vocab.pkl"), "wb") as f:
        pkl.dump(vocab, f)
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_nn_estimator", type=bool, default=True)
    parser.add_argument("--load_lookup_vocabulary", type=bool, default=False)
    parser.add_argument("--load_vocabulary", type=bool, default=True)

    args = parser.parse_args()
    main(args)
