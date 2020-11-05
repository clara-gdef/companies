import os
import argparse
import pickle as pkl
import fastText
import yaml
from tqdm import tqdm
import ipdb
import numpy as np
from nltk.tokenize import word_tokenize


def main(args):
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    with ipdb.launch_ipdb_on_exception():
        with open(os.path.join(CFG["gpudatadir"], "cora_classes_dict.pkl"), 'rb') as f:
            classes = pkl.load(f)

        rev_class_dict = {v: k for k, v in classes.items()}
        print("Loading word vectors...")
        if args.ft_type == "fs":
            embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_fs_cora.bin"))
        else:
            embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_en.bin"))
        print("Word vectors loaded.")

        sent_mean, sent_std, prof_mean, prof_std = build_for_train(embedder, rev_class_dict)

        for split in ['VALID', "TEST"]:
            split_dataset = []
            paper_file = os.path.join(CFG["gpudatadir"], "cora_" + split + ".pkl")
            with open(paper_file, 'rb') as fp:
                data = pkl.load(fp)
            for paper in tqdm(data, desc="Parsing articles for split " + split + "..."):
                identifier = paper[0]
                profile = paper[1]["Abstract"]
                sentence_list = split_abstract_to_sentences(profile)
                if len(sentence_list) > 0:
                    embedded_sentence_list = sentence_list_to_emb(sentence_list, embedder)
                    profile_emb = to_avg_emb(embedded_sentence_list)
                    new_person = {"id": identifier,
                                  "class": rev_class_dict[paper[1]["class"]],
                                  "avg_profile": (profile_emb - prof_mean) / sent_mean,
                                  "sentences_emb": (embedded_sentence_list - sent_mean) /sent_std}
                    split_dataset.append(new_person)
            tgt_file = os.path.join(CFG["gpudatadir"], "cora_embedded_" + args.ft_type + "_" + split + ".pkl")
            with open(tgt_file, 'wb') as f:
                pkl.dump(split_dataset, f)


def split_abstract_to_sentences(profile):
    sentence_list = []
    sentences = " ".join(profile).split(".")
    for sent in sentences:
        # check that sent is a non empty list
        if len(sent) > 0:
            sentence_list.append(word_tokenize(sent))
    return sentence_list


def sentence_list_to_emb(sentence_list, embedder):
    embedded_sentences = []
    for sent in sentence_list:
        if len(sent) > 0:
            tmp = []
            for word in sent:
                tmp.append(embedder.get_word_vector(word))
            embedded_sentences.append(np.mean(np.stack(tmp), axis=0))
    return embedded_sentences


def to_avg_emb(embedded_sentence_list):
    return np.mean(np.stack(embedded_sentence_list), axis=0)


def build_for_train(embedder, rev_class_dict):
    train_dataset = []
    paper_file = os.path.join(CFG["gpudatadir"], "cora_TRAIN.pkl")
    with open(paper_file, 'rb') as fp:
        data = pkl.load(fp)

    all_sentences = np.zeros(300)
    all_profiles = np.zeros(300)

    with ipdb.launch_ipdb_on_exception():
        for paper in tqdm(data, desc="Parsing articles for split TRAIN to get distribution parameters..."):
                profile = paper[1]["Abstract"]
                sentence_list = split_abstract_to_sentences(profile)
                if len(sentence_list) > 0:
                    embedded_sentence_list = sentence_list_to_emb(sentence_list, embedder)
                    for sent in embedded_sentence_list:
                        all_sentences = np.concatenate((all_sentences, sent), axis=0)
                    profile_emb = to_avg_emb(embedded_sentence_list)
                    all_profiles = np.concatenate((all_profiles, profile_emb), axis=0)
        sent_tmp = all_sentences[1:]
        sent_mean = np.mean(sent_tmp, axis=0)
        sent_std = np.std(sent_tmp, axis=0)
        prof_tmp = all_profiles[1:]
        prof_mean = np.mean(prof_tmp, axis=0)
        prof_std = np.std(prof_tmp, axis=0)

        del all_sentences
        del all_profiles

        for paper in tqdm(data, desc="Parsing articles for split TRAIN..."):
            identifier = paper[0]
            profile = paper[1]["Abstract"]
            sentence_list = split_abstract_to_sentences(profile)
            if len(sentence_list) > 0:
                embedded_sentence_list = sentence_list_to_emb(sentence_list, embedder)
                std_emb_sent = (embedded_sentence_list - sent_mean) / sent_std
                profile_emb = ((to_avg_emb(embedded_sentence_list) - prof_mean) / prof_std)
                new_person = {"id": identifier,
                              "class": rev_class_dict[paper[1]["class"]],
                              "avg_profile": profile_emb,
                              "sentences_emb": std_emb_sent}
                train_dataset.append(new_person)

        tgt_file = os.path.join(CFG["gpudatadir"], "cora_embedded_" + args.ft_type + "_TRAIN.pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(train_dataset, f)

    return sent_mean, sent_std, prof_mean, prof_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default="fs")
    args = parser.parse_args()
    main(args)
