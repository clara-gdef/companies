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

num_cie = 207
num_clus = 30
num_dpt = 5888


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        # Load datasets
        xp_title = "disc_spe_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size)
        print(hparams.auto_lr_find)
        datasets = load_datasets(hparams, ["TRAIN", "VALID"])
        dataset_train, dataset_valid = datasets[0], datasets[1]

        # initiate model
        in_size, out_size = dataset_train.get_num_bag(), dataset_train.get_num_bag()
        train_loader = DataLoader(dataset_train, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                                  num_workers=16, shuffle=True)
        valid_loader = DataLoader(dataset_valid, batch_size=hparams.b_size, collate_fn=collate_for_disc_spe_model,
                                  num_workers=16)
        arguments = {'in_size': in_size,
                     'out_size': out_size,
                     'hparams': hparams,
                     'dataset': dataset_train,
                     'datadir': CFG["gpudatadir"],
                     'desc': xp_title}

        print("Initiating model with params (" + str(in_size) + ", " + str(out_size) + ")")
        model = DebugClassifierDisc(**arguments)

        # set up file writers
        log_path = "models/logs/DEBUG/" + hparams.rep_type + "/" + xp_title
        train_writer = SummaryWriter(log_path + "_train", flush_secs=30)
        valid_writer = SummaryWriter(log_path + "_valid", flush_secs=30)
        critetion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=hparams.lr)

        for epoch in range(1, hparams.epochs + 1):
            dico = main_for_one_epoch(hparams, epoch, model, optim, critetion,
                                      best_val_loss, train_loader, valid_loader, train_writer, valid_writer)
            best_val_loss = dico['best_val_loss']


def main_for_one_epoch(hparams, epoch, model, optim, critetion,
                                      best_val_loss, train_loader, valid_loader,
                                    train_writer, valid_writer):
    print("Training and validating for epoch " + str(epoch))

    train_loss = train(hparams, train_loader, model, optim, critetion, epoch)

    for k, v in train_loss.items():
        train_writer.add_scalar(k, v, epoch)

    valid_loss = valid(hparams, valid_loader, model, critetion, epoch)

    for k, v in valid_loss.items():
        valid_writer.add_scalar(k, v, epoch)

    file_name = "debug_disc_spe_" + str(hparams.rep_type) + "_" + str(hparams.bag_type) + "_" + str(hparams.lr)
    file_path = os.path.join(CFG["modeldir"], file_name)
    if valid_loss['CE'] < best_val_loss:
        torch.save(model.state_dict, file_path)
        best_val_loss = valid_loss['CE']

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def train(hparams, train_loader, model, crit, optim):
    loss_list = []
    for ids, ppl, labels, bag_rep in tqdm(train_loader):
        bag_rep = torch.transpose(bag_rep, 1, 0)
        input_tensor = torch.matmul(ppl, bag_rep)
        tmp_labels = get_labels(labels, hparams.bag_type)
        labels = torch.LongTensor(tmp_labels).view(output.shape[0]).cuda()
        output = model(input_tensor)
        # the model is specialized
        loss = crit(torch.softmax(output, dim=1), labels)
        loss_list.append(loss)
        loss.backward()
        optim.step()

    return {'CE': torch.mean(torch.stack(loss_list)).item()}


def valid(hparams, valid_loader, model, crit):
    loss_list = []
    for ids, ppl, labels, bag_rep in tqdm(valid_loader):
        bag_rep = torch.transpose(bag_rep, 1, 0)
        input_tensor = torch.matmul(ppl, bag_rep)
        tmp_labels = get_labels(labels, hparams.bag_type)
        labels = torch.LongTensor(tmp_labels).view(output.shape[0]).cuda()
        output = model(input_tensor)
        # the model is specialized
        loss = crit(torch.softmax(output, dim=1), labels)
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


def get_labels(batch, bag_type):
    if bag_type == "cie":
        tmp_labels = [batch[2]]
    elif bag_type == "clus":
        offset = num_cie
        tmp_labels = [[i - offset for i in batch[2]]]
    elif bag_type == "dpt":
        offset = num_cie + num_clus
        tmp_labels = [[i - offset for i in batch[2]]]
    else:
        raise Exception("Wrong bag type specified: " + str(bag_type))
    return tmp_labels


if __name__ == "__main__":
    global CFG
    with open("config.yaml", "r") as ymlfile:
        CFG = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep_type", type=str, default='ft')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--b_size", type=int, default=2)
    parser.add_argument("--input_type", type=str, default="matMul")
    parser.add_argument("--data_agg_type", type=str, default="avg")
    parser.add_argument("--bag_type", type=str, default="cie")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--auto_lr_find", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    hparams = parser.parse_args()
    main(hparams)

