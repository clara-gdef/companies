import ipdb
import yaml
import os
import pickle as pkl
from models.baselines import cora_bow_svm
from utils import DotDict


def main():
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    high_level = True
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

    with ipdb.launch_ipdb_set_trace():
        results = {}
        dico = {}
        for min_df in [1e-2, 1e-3, 1e-4]:
            results[min_df] = {}
            for max_df in [.7, .8, .9]:
                results[min_df][max_df] = {}
                for max_voc_size in [3e4, 4e4, 5e4]:
                    results[min_df][max_df] = {}
                    dico['min_df'] = min_df
                    dico['max_df'] = max_df
                    dico['max_voc_size'] = max_voc_size
                    arg = DotDict(dico)
                    results[min_df][max_df][max_voc_size] = cora_bow_svm.main(arg, data_train, data_test, rev_class_dict, mapper_dict, class_dict, high_level)
        ipdb.set_trace()


def analyze_results(results):
    ipdb.set_trace()

if __name__ == "__main__":
    main()
