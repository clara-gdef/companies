import os
import torch

import ipdb
import argparse
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data.datasets import DiscriminativeSpecializedDataset
from models.classes import DebugClassifierDisc
from utils.models import collate_for_disc_spe_model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

num_cie = 207
num_clus = 30
num_dpt = 5888


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        # Load datasets
        xp_title = "disc_spe_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size) + "_lr" + str(hparams.lr)
        datasets = load_datasets(hparams, ["TRAIN", "VALID"])
        dataset_train, dataset_valid = datasets[0], datasets[1]

        # initiate model
        best_val_loss = 1e+300
        in_size, out_size = dataset_train.get_num_bag(), dataset_train.get_num_bag()
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                                  num_workers=0, shuffle=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                                  num_workers=0)
        arguments = {'in_size': in_size,
                     'out_size': out_size,
                     'hparams': hparams,
                     'dataset': dataset_train,
                     'datadir': CFG["gpudatadir"],
                     'desc': xp_title,
                     "constant_weight": 1}

        print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
        model = DebugClassifierDisc(**arguments)

        # set up file writers
        log_path = "models/logs/DEBUG/" + hparams.rep_type + "/" + xp_title
        train_writer = SummaryWriter(log_path + "_train", flush_secs=1)
        valid_writer = SummaryWriter(log_path + "_valid", flush_secs=1)
        critetion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=hparams.lr)

        for epoch in range(1, hparams.epochs + 1):
            dico = main_for_one_epoch(epoch, model.cuda(), optim, critetion,
                                      best_val_loss, train_loader, valid_loader, train_writer, valid_writer, xp_title)
            best_val_loss = dico['best_val_loss']


def main_for_one_epoch(epoch, model, optim, critetion,
                       best_val_loss, train_loader, valid_loader,
                       train_writer, valid_writer, xp_title):
    print("Training and validating for epoch " + str(epoch))

    train_loss = train(train_loader, model, critetion, optim, epoch)

    for k, v in train_loss.items():
        train_writer.add_scalar(k, v, epoch)

    valid_loss = valid(valid_loader, model, critetion, epoch)

    for k, v in valid_loss.items():
        valid_writer.add_scalar(k, v, epoch)

    file_path = os.path.join(CFG["modeldir"] + "/DEBUG", xp_title)
    if valid_loss['CE'] < best_val_loss:
        torch.save(model.state_dict(), file_path)
        best_val_loss = valid_loss['CE']

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(train_loader, model, crit, optim, epoch):
    loss_list = []
    b4_training = []
    labs = []
    preds = []
    num_classes = num_cie
    for ids, ppl, tmp_labels, bag_rep in tqdm(train_loader, desc="Training for epoch " + str(
            epoch) + "..."):
        bag_rep = torch.transpose(bag_rep, 1, 0)
        input_tensor = torch.matmul(ppl, bag_rep).cuda()
        output = model(input_tensor)
        labels = torch.LongTensor(tmp_labels).view(output.shape[0]).cuda()
        # the model is specialized
        loss = crit(output, labels)
        loss_list.append(loss)
        loss.backward()
        optim.step()

        ipdb.set_trace()

        b4_training.extend(torch.argmax(input_tensor, dim=1))
        preds.extend(torch.argmax(output, dim=1))
        labs.extend(tmp_labels)

    res_dict = {"acc_trained": accuracy_score(preds, labels) * 100,
                "acc_b4_training": accuracy_score(b4_training, labels) * 100,
                "precision_trained": precision_score(preds, labels, average='weighted',
                                                     labels=range(num_classes)) * 100,
                "precision_b4_training": precision_score(b4_training, labels, average='weighted',
                                                         labels=range(num_classes)) * 100,
                "recall_trained": recall_score(preds, labels, average='weighted', labels=range(num_classes)) * 100,
                "recall_b4_training": recall_score(b4_training, labels, average='weighted',
                                                   labels=range(num_classes)) * 100,
                "f1_trained": f1_score(preds, labels, average='weighted', labels=range(num_classes)) * 100,
                "f1_b4_training": f1_score(b4_training, labels, average='weighted', labels=range(num_classes)) * 100,
                'CE': torch.mean(torch.stack(loss_list)).item()}

    return res_dict


def valid(valid_loader, model, crit, epoch):
    loss_list = []
    for ids, ppl, tmp_labels, bag_rep in tqdm(valid_loader, desc="Validating for epoch " + str(
            epoch) + "..."):
        bag_rep = torch.transpose(bag_rep, 1, 0)
        input_tensor = torch.matmul(ppl, bag_rep).cuda()
        output = model(input_tensor)
        labels = torch.LongTensor(tmp_labels).view(output.shape[0]).cuda()
        # the model is specialized
        loss = crit(output, labels)
        loss_list.append(loss)

    return {'CE': torch.mean(torch.stack(loss_list)).item()}


def load_datasets(hparams, splits):
    datasets = []
    common_hparams = {
        "data_dir": CFG["gpudatadir"],
        "rep_type": hparams.rep_type,
        "agg_type": hparams.data_agg_type,
        "bag_type": hparams.bag_type,
    }
    for split in splits:
        datasets.append(DiscriminativeSpecializedDataset(**common_hparams, split=split))

    return datasets


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=64)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    hparams = parser.parse_args()
    main(hparams)
