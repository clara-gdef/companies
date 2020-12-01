import ipdb
import yaml
import os
import pickle as pkl
from models.baselines import cora_bow_svm
from utils import DotDict
from collections import Counter


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

    class_weights = get_class_weights(data_train, rev_class_dict, mapper_dict)

    with ipdb.launch_ipdb_on_exception():
        results = {}
        dico = {}
        for min_df in [1, 1e-2, 1e-3, 1e-4]:
            results[min_df] = {}
            for max_df in [.7, .8, .9, .1]:
                results[min_df][max_df] = {}
                for max_voc_size in [1000, 1150, 1300]:
                    results[min_df][max_df] = {}
                    dico['min_df'] = min_df
                    dico['max_df'] = max_df
                    dico['max_voc_size'] = int(max_voc_size)
                    arg = DotDict(dico)
                    results[min_df][max_df][int(max_voc_size)] = cora_bow_svm.main(arg, data_train, data_test, rev_class_dict, mapper_dict, class_dict, high_level, class_weights)
        best_acc_keys, best_f1_keys = analyze_results(results, 'SVM_BOW')
        print("FOR BEST ACCURACY: " +  str(best_acc_keys))
        print("RESULTS: " + str(results[best_acc_keys[0]][best_acc_keys[1]][best_acc_keys[2]]))
        print("FOR BEST F1: " +  str(best_f1_keys))
        print("RESULTS: " + str(results[best_f1_keys[0]][best_f1_keys[1]][best_f1_keys[2]]))
        ipdb.set_trace()


def get_class_weights(data_train, rev_class_dict, mapper_dict):
    class_counter = Counter()
    for item in data_train:
        class_counter[mapper_dict[rev_class_dict[item[1]["class"]]]] +=1
    total_samples = sum([i for i in class_counter.values()])
    class_weight = {k: v/total_samples for k, v in class_counter.items()}
    return class_weight

def analyze_results(test_results, handle):
    best_acc = 0
    best_f1 = 0
    best_acc_keys = None
    best_f1_keys = None
    for min_df in test_results.keys():
        for max_df in test_results[min_df].keys():
            for max_voc_size in test_results[min_df][max_df].keys():
                if test_results[min_df][max_df][max_voc_size]["acc_" + handle] > best_acc:
                    best_acc_keys = (min_df, max_df, max_voc_size)
                    best_acc = test_results[min_df][max_df][max_voc_size]["acc_" + handle]
                if test_results[min_df][max_df][max_voc_size]["f1_" + handle] > best_f1:
                    best_f1_keys = (min_df, max_df, max_voc_size)
                    best_f1 = test_results[min_df][max_df][max_voc_size]["f1_" + handle]
    print("Evaluated for min_df= [" + str(test_results.keys()) + "], max_df=[" + str(test_results[min_df].keys()) + "]")
    return best_acc_keys, best_f1_keys


if __name__ == "__main__":
    main()
