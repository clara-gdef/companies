import ipdb
import torch
import numpy as np


def collate_for_disc_poly_model(batch):
    ids = [i[0]["id"] for i in batch]
    reps = [i[0]["rep"] for i in batch]
    cies = [i[0]["cie"] for i in batch]
    clus = [i[0]["clus"] for i in batch]
    dpt = [i[0]["dpt"] for i in batch]
    return ids, torch.stack(reps), cies, clus, dpt, batch[0][-1]


def collate_for_attn_disc_poly_model(batch):
    ids = [i[0] for i in batch]
    max_profil_len = max([len(i[1]) for i in batch])
    reps = torch.FloatTensor(len(batch), max_profil_len, 300)
    for idx, person in enumerate(batch):
        reps[idx][:len(person[1])][:] = torch.from_numpy(person[1]).type(torch.FloatTensor).cuda()
    cies = [i[2] for i in batch]
    clus = [i[3] for i in batch]
    dpt = [i[4] for i in batch]
    return ids, reps, cies, clus, dpt, batch[0][-1]


def collate_for_gen_poly_model(batch):
    ppl = [i[0] for i in batch]
    bag_reps = [i[1] for i in batch]
    labels = [i[-1] for i in batch]
    return torch.stack(ppl), torch.stack(bag_reps), labels


def collate_for_disc_spe_model(batch):
    ids = [i["id"] for i in batch]
    ppl = [i["ppl_rep"] for i in batch]
    labels = [i["label"] for i in batch]
    return ids, torch.stack(ppl), labels, batch[0]["bag_rep"]


def labels_to_one_hot(b_size, classes_num, total_classes):
    one_hot = torch.zeros(b_size, total_classes)
    for labels in classes_num:
        for batch_num, class_num in enumerate(labels):
            one_hot[batch_num][class_num] = 1.
    return one_hot


def collate_for_jobs(batch):
    ids = [i[0] for i in batch]
    jobs = [i[1] for i in batch]
    jobs_emb = [i[2] for i in batch]
    pred = [i[3] for i in batch]
    label = [i[4] for i in batch]
    return ids, torch.FloatTensor(np.stack(jobs_emb)[0]), label, torch.FloatTensor(np.stack(pred)), jobs,  batch[0][-1]


def class_to_one_hot(num_ppl, classes_num, total_classes):
    one_hot = torch.zeros(num_ppl, total_classes)
    for person_num, class_num in enumerate(classes_num):
        one_hot[person_num][class_num] = 1.
    return one_hot.cuda()


def get_model_params(hparams, rep_dim, num_bag):
    out_size = num_bag
    if hparams.input_type == "concat":
        in_size = rep_dim * num_bag
    elif hparams.input_type == "matMul" or hparams.input_type == "hadamard":
        in_size = num_bag
    elif hparams.input_type == "userOriented" or hparams.input_type == "bagTransformer":
        in_size = rep_dim
        out_size = rep_dim
    elif hparams.input_type == "userOnly":
        in_size = rep_dim
    elif hparams.input_type == "b4Training":
        in_size = num_bag
    else:
        raise Exception("Wrong input data specified: " + str(hparams.input_type))
    return in_size, out_size

