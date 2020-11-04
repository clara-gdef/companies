import os
import argparse
import pickle as pkl
import fastText
import yaml
from tqdm import tqdm
import ipdb
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
            embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_cora.bin"))
        else:
            embedder = fastText.load_model(os.path.join(CFG["modeldir"], "ft_en.bin"))
        print("Word vectors loaded.")

        for split in ["TRAIN", 'VALID', "TEST"]:
            split_dataset = []
            paper_file = os.path.join(CFG["gpudatadir"], "cora_" + split + ".pkl")
            with open(paper_file, 'rb') as fp:
                data = pkl.load(fp)
            for paper in tqdm(data, desc="Parsing articles for split " + split + "..."):
                identifier = paper[0]
                profile = paper[1]["Abstract"]
                sentence_list = split_abstract_to_sentences(profile)
                embedded_sentence_list = sentence_list_to_emb(sentence_list, embedder)
                profile_emb = to_avg_emb(embedded_sentence_list)
                new_person = {"id": identifier,
                              "class": rev_class_dict[paper[1]["class"]],
                              "avg_profile": profile_emb,
                              "sentences_emb": embedded_sentence_list}
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
    ipdb.set_trace()

def to_avg_emb(embedded_sentence_list):
    ipdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_type", type=str, default="fs")
    args = parser.parse_args()
    main(args)
