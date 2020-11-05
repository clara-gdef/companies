import itertools

import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import pickle as pkl
from tqdm import tqdm
from utils.models import labels_to_one_hot
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


class InstanceClassifierDiscCora(pl.LightningModule):
    def __init__(self, in_size, out_size, hparams, input_type, ft_type, num_tracks, datadir, desc):
        super().__init__()
        self.input_type = hparams.input_type
        self.num_tracks = num_tracks
        self.ft_type = ft_type
        self.input_type = input_type

        self.hp = hparams
        self.data_dir = datadir
        self.description = desc

        if self.input_type == "hadamard":
            self.lin_dim_reduction = torch.nn.Linear(in_size, self.hp.middle_size)
            self.lin_class_prediction = torch.nn.Linear(self.hp.middle_size, out_size)
            torch.nn.init.eye_(self.lin_dim_reduction.weight)
            torch.nn.init.eye_(self.lin_class_prediction.weight)
            torch.nn.init.zeros_(self.lin_dim_reduction.bias)
            torch.nn.init.zeros_(self.lin_class_prediction.bias)
        else:
            self.lin = torch.nn.Linear(in_size, out_size)
            torch.nn.init.eye_(self.lin.weight)
            torch.nn.init.zeros_(self.lin.bias)

        self.training_losses = []
        self.test_outputs = []
        self.test_labels_one_hot = []
        self.test_labels = []
        self.test_ppl_id = []

        ### debug
        self.before_training = []

    def forward(self, x):
        if self.input_type == "bagTransformer":
            mat = torch.diag(self.lin.weight).unsqueeze(1)
            out = x * mat + self.lin.bias.view(x.shape[0], -1)
            return out.T
        elif self.input_type == "hadamard":
            transformed_input = torch.relu(self.lin_dim_reduction(x))
            return self.lin_class_prediction(transformed_input)
        else:
            return self.lin(x)

    def training_step(self, batch, batch_nb):
        labels = batch[-2]
        if self.input_type != "userOriented" and self.input_type != "bagTransformer":
            input_tensor = self.get_input_tensor(batch)
            output = self.forward(input_tensor)
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            if self.input_type == "userOriented":
                tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
            if self.input_type == "bagTransformer":
                new_bags = self(bag_matrix.T)
                tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
        loss = torch.nn.functional.cross_entropy(output, labels)
        tensorboard_logs = {'train_loss': loss}
        self.training_losses.append(loss.item())
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        labels = batch[-2]
        if self.input_type != "userOriented" and self.input_type != "bagTransformer":
            input_tensor = self.get_input_tensor(batch)
            output = self.forward(input_tensor)
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            if self.input_type == "userOriented":
                tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
            if self.input_type == "bagTransformer":
                new_bags = self(bag_matrix.T)
                tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
        val_loss = torch.nn.functional.cross_entropy(output, labels)
        tensorboard_logs = {'val_loss': val_loss}
        return {'loss': val_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        return outputs[-1]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.wd)

    def test_step(self, batch, batch_idx):
        if self.input_type == "userOriented":
            bag_matrix, profiles = self.get_input_tensor(batch)
            labels = batch[-2]
            self.test_outputs.append(torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0)).cuda())
        elif self.input_type == "b4Training":
            input_tensor = self.get_input_tensor(batch)
            self.test_outputs.append(input_tensor)
            labels = batch[-2]
        elif self.input_type == "bagTransformer":
            bag_matrix, profiles = self.get_input_tensor(batch)
            labels = batch[-2]
            new_bags = self(bag_matrix.T)
            tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
            self.test_outputs.append(torch.transpose(tmp, 1, 0))
        else:
            input_tensor = self.get_input_tensor(batch)
            labels = batch[-2]
            self.test_outputs.append(self.forward(input_tensor))
            self.before_training.append(input_tensor)
        self.test_labels.append(labels)
        self.test_ppl_id.append(batch[0])

    def test_epoch_end(self, outputs):
        outputs = torch.stack(self.test_outputs)
        return self.test_spe(outputs)

    def test_spe(self, outputs):
        preds = torch.argsort(outputs.view(-1, self.num_tracks), dim=-1, descending=True)
        labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
        res_dict_trained = get_metrics(preds[:, :1].cpu(), labels.cpu(), self.num_tracks, self.bag_type, 0)
        for k in [10]:
            tmp = get_metrics_at_k(preds[:, :k].cpu(), labels.cpu(), self.num_tracks,
                                                   self.bag_type + "_@" + str(k), 0)
            res_dict_trained = {**res_dict_trained, **tmp}
        self.save_bag_outputs(preds, labels, confusion_matrix(preds[:, :1].cpu(), labels.cpu()), res_dict_trained)
        return res_dict_trained

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


    def get_input_tensor(self, batch):
        profiles = batch[1]
        if self.input_type == "matMul" or self.input_type == "b4Training" or self.input_type == "hadamard":
            bag_rep = batch[-1].T
            input_tensor = torch.matmul(profiles, bag_rep)
        elif self.input_type == "concat":
            raise NotImplementedError
        elif self.input_type == "userOriented":
            input_tensor = (batch[-1], profiles)
        elif self.input_type == "bagTransformer":
            input_tensor = (batch[-1], profiles)
        elif self.input_type == "userOnly":
            input_tensor = profiles

        else:
            raise Exception("Wrong input data specified: " + str(self.input_type))
        return input_tensor

    def get_outputs_and_labels(self, test_loader):
        idx = []
        for batch in tqdm(test_loader, desc="Testing..."):
            idx.append(batch[0][0])
            self.test_step(batch, 0)
        outputs = torch.stack(self.test_outputs)
        if self.type == "poly":
            if self.input_type != "userOriented":
                cie_preds = outputs[:, 0, :self.num_cie]
                clus_preds = outputs[:, 0, self.num_cie: self.num_cie + self.num_clus]
                dpt_preds = outputs[:, 0, -self.num_dpt:]
            else:
                cie_preds = outputs[:, :self.num_cie, 0]
                clus_preds = outputs[:, self.num_cie: self.num_cie + self.num_clus, 0]
                dpt_preds = outputs[:, -self.num_dpt:, 0]

            cie_labels = torch.LongTensor([i[0][0] for i in self.test_labels])
            clus_labels = torch.LongTensor([i[1][0] for i in self.test_labels])
            dpt_labels = torch.LongTensor([i[2][0] for i in self.test_labels])

            return {"indices": idx,
                    "preds":
                        {"cie": cie_preds,
                         "clus": clus_preds,
                         "dpt": dpt_preds},
                    "labels":
                        {"cie": cie_labels,
                         "clus": clus_labels,
                         "dpt": dpt_labels},
                    }
        else:
            if self.bag_type == "cie":
                offset = 0
                limit = self.num_cie
            elif self.bag_type == "clus":
                offset = self.num_cie
                limit = self.num_cie + self.num_clus
            elif self.bag_type == "dpt":
                offset = self.num_cie + self.num_clus
                limit = self.num_cie  + self.num_clus + self.num_dpt
            if self.input_type != "userOriented":
                preds = outputs[:, 0, offset:limit]
            else:
                preds = outputs[:, offset:limit, 0]
            labels = torch.LongTensor([i[0][0] for i in self.test_labels])
            return {"indices": idx,
                    "preds": preds,
                    "labels": labels
                    }

    def get_jobs_outputs(self, test_loader):
        jobs_ouputs = {}
        for batch in tqdm(test_loader, desc="Testing..."):
            jobs_ouputs[batch[0][0]] = {}
            jobs_ouputs[batch[0][0]]["jobs"] = batch[-2]
            jobs_ouputs[batch[0][0]]["jobs_emb"] = batch[1]
            self.test_step(batch, 0)
        counter = 0
        for batch in tqdm(test_loader, desc="Testing..."):
            jobs_ouputs[batch[0][0]]["jobs_outputs"] = self.test_outputs[counter]
            jobs_ouputs[batch[0][0]]["labels"] = self.test_labels[counter]
            counter += 1
        return jobs_ouputs


def test_for_all_bags(cie_labels, clus_labels, dpt_labels, cie_preds, clus_preds, dpt_preds, num_classes):
    all_labels = []
    for tup in zip(cie_labels, clus_labels, dpt_labels):
        all_labels.append([tup[0].item(), tup[1].item(), tup[2].item()])
    cie_preds_max = [i.item() for i in torch.argmax(cie_preds, dim=1)]
    clus_preds_max = [i.item() + 207 for i in torch.argmax(clus_preds, dim=1)]
    dpt_preds_max = [i.item() + 237 for i in torch.argmax(dpt_preds, dim=1)]
    all_preds_max = []
    for tup in zip(cie_preds_max, clus_preds_max, dpt_preds_max):
        all_preds_max.append([tup[0], tup[1], tup[2]])

    general_res = get_metrics(np.array(all_preds_max).reshape(-1, 1), np.array(all_labels).reshape(-1, 1), num_classes,
                              "all", 0)
    cie_preds_at_k = [i for i in torch.argsort(cie_preds, dim=-1, descending=True)]
    clus_preds_at_k = [i + 207 for i in torch.argsort(clus_preds, dim=-1, descending=True)]
    dpt_preds_at_k = [i + 237 for i in torch.argsort(dpt_preds, dim=-1, descending=True)]

    all_preds_k = []
    chained_labels = [i for i in itertools.chain(cie_labels, clus_labels, dpt_labels)]
    for preds, labels in tqdm(zip(itertools.chain(cie_preds_at_k, clus_preds_at_k, dpt_preds_at_k), chained_labels), desc="Computing at k=10..."):
        if labels.item() in preds[:10]:
            all_preds_k.append(labels.item())
        else:
            if type(preds) == torch.Tensor:
                all_preds_k.append(preds[0].item())
            else:
                all_preds_k.append(preds)

    res_at_k = get_metrics(all_preds_k, chained_labels, num_classes, "all_@k", 0)

    return {**general_res, **res_at_k}


def test_for_bag(preds, labels, b4_training, offset, num_classes, bag_type):
    predicted_classes = torch.argsort(preds, dim=-1, descending=True)
    res_dict_trained = get_metrics([i.item() + offset for i in predicted_classes[:, 0]], labels.cpu(), num_classes,
                                      bag_type, offset)

    for k in [10]:
        tmp = get_metrics_at_k(predicted_classes[:, :k].cpu(), labels.cpu(), num_classes,
                                               bag_type + "_@" + str(k), offset)
        res_dict_trained = {**res_dict_trained, **tmp}
    return res_dict_trained


def get_average_metrics(res_dict):
    precision = []
    recall = []
    numerical_keys = [i for i in res_dict.keys()][:-3]
    for k in numerical_keys:
        precision.append(res_dict[k]["precision"])
        recall.append(res_dict[k]["recall"])
    return np.mean(precision), np.mean(recall)


def get_metrics(preds, labels, num_classes, handle, offset):
    num_c = range(offset, offset + num_classes)
    res_dict = {
        "acc_" + handle: accuracy_score(labels, preds) * 100,
        "precision_" + handle: precision_score(labels, preds, average='weighted',
                                               labels=num_c, zero_division=0) * 100,
        "recall_" + handle: recall_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100,
        "f1_" + handle: f1_score(labels, preds, average='weighted', labels=num_c, zero_division=0) * 100}
    return res_dict


def get_metrics_at_k(predictions, labels, num_classes, handle, offset):
    out_predictions = []
    transformed_predictions = predictions + offset
    for index, pred in enumerate(transformed_predictions):
        if labels[index].item() in pred:
            out_predictions.append(labels[index].item())
        else:
            if type(pred[0]) == torch.Tensor:
                out_predictions.append(pred[0].item())
            else:
                out_predictions.append(pred[0])
    return get_metrics(out_predictions, labels, num_classes, handle, offset)
