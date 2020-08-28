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


def main(hparams):
    with ipdb.launch_ipdb_on_exception():
        # Load datasets
        xp_title = "disc_spe_" + hparams.bag_type + "_" + hparams.rep_type + "_" + hparams.data_agg_type + "_" + hparams.input_type + "_bs" + str(
            hparams.b_size)
        print(hparams.auto_lr_find)
        datasets = load_datasets(hparams, ["TEST"])
        dataset_test = datasets[0]

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

    if valid_loss['overallLoss'] < best_val_loss:
        torch.save(model.state_dict, )
        best_val_loss = valid_loss['overallLoss']

    dictionary = {**train_loss, **valid_loss, 'best_val_loss': best_val_loss}
    return dictionary


def test(self, batch, batch_idx):
    if self.input_type != "userOriented":
        input_tensor = self.get_input_tensor(batch)
        tmp_labels = self.get_labels(batch)
        labels_one_hot = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
        self.test_outputs.append(self.forward(input_tensor))
        self.before_training.append(input_tensor)
        ##### B4 training #####
        # self.test_outputs.append(input_tensor)
    self.test_labels_one_hot.append(labels_one_hot)
    self.test_labels.append(tmp_labels)
    self.test_ppl_id.append(batch[0])


def test_epoch_end(self, outputs):
    outputs = torch.stack(self.test_outputs)
    if self.type == "poly":
        res = self.test_poly(outputs)
    else:
        res = self.test_spe(outputs)
    return res


def test_poly(self, outputs):
    # slicing predictions per class type
    if self.input_type != "userOriented":
        cie_preds = outputs[:, 0, :self.num_cie]
        clus_preds = outputs[:, 0, self.num_cie: self.num_cie + self.num_clus]
        dpt_preds = outputs[:, 0, -self.num_dpt:]
    else:
        cie_preds = outputs[:, :self.num_cie, 0]
        clus_preds = outputs[:, self.num_cie: self.num_cie + self.num_clus, 0]
        dpt_preds = outputs[:, -self.num_dpt:, 0]

    cie_labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
    clus_labels = torch.LongTensor([i[1][0] for i in self.test_labels]).cuda()
    dpt_labels = torch.LongTensor([i[2][0] for i in self.test_labels]).cuda()

    ci_preds, ci_cm, ci_res = test_for_bag(cie_preds, cie_labels, offset=0)
    cl_preds, cl_cm, cl_res = test_for_bag(clus_preds, clus_labels, offset=self.num_cie)
    d_preds, d_cm, d_res = test_for_bag(dpt_preds, dpt_labels, offset=self.num_cie + self.num_clus)
    # self.save_outputs(ci_preds, cie_labels, ci_cm, ci_res,
    #                   cl_preds, clus_labels, cl_cm, cl_res,
    #                   d_preds, dpt_labels, d_cm, d_res)

    ci_avg_prec, ci_avg_rec = get_average_metrics(ci_res)
    cl_avg_prec, cl_avg_rec = get_average_metrics(cl_res)
    d_avg_prec, d_avg_rec = get_average_metrics(d_res)
    return {"cie_acc": ci_res["accuracy"],
            "cie_precision": ci_avg_prec,
            "cie_recall": ci_avg_rec,
            "clus_acc": cl_res["accuracy"],
            "clus_precision": cl_avg_prec,
            "clus_recall": cl_avg_rec,
            "dpt_acc": d_res["accuracy"],
            "dpt_precision": d_avg_prec,
            "dpt_recall": d_avg_rec
            }


def test_spe(self, outputs):
    ipdb.set_trace()
    preds = outputs[:, 0, :]
    labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
    preds, cm, res = test_for_bag(preds, labels, self.before_training, 0, self.get_num_classes())
    self.save_bag_outputs(preds, labels, cm, res)
    prec, rec = get_average_metrics(res)
    return {"acc": res["accuracy"],
            "precision": prec,
            "recall": rec}

    def save_bag_outputs(self, preds, labels, cm, res):
        res = {self.bag_type: {"preds": preds,
                               "labels": labels,
                               "cm": cm,
                               "res": res}
               }
        tgt_file = os.path.join(self.data_dir, "OUTPUTS_" + self.description + ".pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(res, f)

    def save_outputs(self, ci_preds, cie_labels, ci_cm, ci_res,
                     cl_preds, clus_labels, cl_cm, cl_res,
                     d_preds, dpt_labels, d_cm, d_res):
        res = {"cie": {"preds": ci_preds,
                       "labels": cie_labels,
                       "cm": ci_cm,
                       "res": ci_res},
               "clus": {"preds": cl_preds,
                        "labels": clus_labels,
                        "cm": cl_cm,
                        "res": cl_res},
               "dpt": {"preds": d_preds,
                       "labels": dpt_labels,
                       "cm": d_cm,
                       "res": d_res},
               "ppl": self.test_ppl_id
               }
        tgt_file = os.path.join(self.data_dir, "OUTPUTS_" + self.description + ".pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(res, f)

    def get_labels(self, batch):
        if self.type == "poly":
            tmp_labels = [batch[2], batch[3], batch[4]]
        elif self.type == "spe":
            if self.bag_type == "cie":
                tmp_labels = [batch[2]]
            elif self.bag_type == "clus":
                offset = self.num_cie
                tmp_labels = [[i - offset for i in batch[2]]]
            elif self.bag_type == "dpt":
                offset = self.num_cie + self.num_clus
                tmp_labels = [[i - offset for i in batch[2]]]
            else:
                raise Exception("Wrong bag type specified: " + str(self.bag_type))
        else:
            raise Exception("Wrong model type specified: " + str(self.type) + ", can be either \"poly\" or \"spe\"")
        return tmp_labels

    def get_input_tensor(self, batch):
        if len(batch[1].shape) > 2:
            profiles = batch[1].squeeze(1)
        else:
            profiles = batch[1]
        if self.input_type == "matMul":
            bag_rep = torch.transpose(batch[-1], 1, 0)
            input_tensor = torch.matmul(profiles, bag_rep)
            ipdb.set_trace()
        elif self.input_type == "concat":
            raise NotImplementedError
        elif self.input_type == "hadamard":
            b_size = profiles.shape[0]
            bag_rep = batch[-1]
            expanded_bag_rep = bag_rep.expand(b_size, bag_rep.shape[0], bag_rep.shape[-1])
            prof = profiles.unsqueeze(1)
            expanded_profiles = prof.expand(b_size, bag_rep.shape[0], bag_rep.shape[-1])
            tmp = expanded_bag_rep * expanded_profiles
            input_tensor = tmp.view(b_size, -1)
        elif self.input_type == "userOriented":
            input_tensor = (batch[-1], profiles)
        elif self.input_type == "userOnly":
            input_tensor = profiles
        else:
            raise Exception("Wrong input data specified: " + str(self.input_type))
        return input_tensor

    def get_num_classes(self):
        if self.type == "spe":
            if self.bag_type == "cie":
                return self.num_cie
            elif self.bag_type == "clus":
                return self.num_clus
            elif self.bag_type == "dpt":
                return self.num_dpt
        else:
            return self.num_cie + self.num_clus + self.num_dpt


def test_for_bag(preds, labels, b4_training, offset, num_classes):
    tmp = [i.item() for i in torch.argmax(preds, dim=1)]
    tmp2 = [i.item() for i in torch.argmax(torch.stack(b4_training)[:, 0, :], dim=1)]
    ipdb.set_trace()
    predicted_classes = [i + offset for i in tmp]
    cm = confusion_matrix(labels.cpu().numpy(), np.asarray(predicted_classes))
    results = classification_report(predicted_classes, labels.cpu(), output_dict=True, labels=range(num_classes))
    return predicted_classes, cm, results


def get_average_metrics(res_dict):
    precision = []
    recall = []
    numerical_keys = [i for i in res_dict.keys()][:-3]
    for k in numerical_keys:
        precision.append(res_dict[k]["precision"])
        recall.append(res_dict[k]["recall"])
    return np.mean(precision), np.mean(recall)


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


def get_labels(self, batch):
    if self.bag_type == "cie":
        tmp_labels = [batch[2]]
    elif self.bag_type == "clus":
        offset = self.num_cie
        tmp_labels = [[i - offset for i in batch[2]]]
    elif self.bag_type == "dpt":
        offset = self.num_cie + self.num_clus
        tmp_labels = [[i - offset for i in batch[2]]]
    else:
        raise Exception("Wrong bag type specified: " + str(self.bag_type))
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

