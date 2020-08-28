import ipdb
import torch


def collate_for_disc_poly_model(batch):
    ids = [i[0]["id"] for i in batch]
    reps = [i[0]["rep"] for i in batch]
    cies = [i[0]["cie"] for i in batch]
    clus = [i[0]["clus"] for i in batch]
    dpt = [i[0]["dpt"] for i in batch]
    return ids, torch.stack(reps), cies, clus, dpt, batch[0][-1]


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


def class_to_one_hot(num_ppl, classes_num, total_classes):
    one_hot = torch.zeros(num_ppl, total_classes)
    for person_num, class_num in enumerate(classes_num):
        one_hot[person_num][class_num] = 1.
    return one_hot.cuda()

