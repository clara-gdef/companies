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


class InstanceClassifierDisc(pl.LightningModule):
    def __init__(self, in_size, out_size, hparams, dataset, datadir, desc, wd, bag_type, middle_size=None):
        super().__init__()
        self.middle_size = middle_size
        self.wd = wd
        self.input_type = hparams.input_type
        self.num_cie = dataset.num_cie
        self.num_clus = dataset.num_clus
        self.num_dpt = dataset.num_dpt
        self.type = desc.split("_")[1]
        if self.type == 'spe':
            self.bag_type = bag_type

        self.hparams = hparams
        self.data_dir = datadir
        self.description = desc

        if self.input_type == "hadamard":
            self.lin_dim_reduction = torch.nn.Linear(in_size, middle_size)
            self.lin_class_prediction = torch.nn.Linear(middle_size, out_size)
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
            # if out.T.shape != (x.shape[1], x.shape[0]):
            #     ipdb.set_trace()
            return out.T
        elif self.input_type == "hadamard":
            transformed_input = torch.relu(self.lin_dim_reduction(x))
            return self.lin_class_prediction(transformed_input)
        else:
            return self.lin(x)

    def training_step(self, batch, batch_nb):
        if self.input_type != "userOriented" and self.input_type != "bagTransformer":
            input_tensor = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            output = self.forward(input_tensor)
            labels = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            if self.input_type == "userOriented":
                tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
            if self.input_type == "bagTransformer":
                new_bags = self(bag_matrix.T)
                tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
                # if self.input_type == "hadamard":
                #
                #     output = self((bag_matrix, profiles))
        if self.type == "poly":
            cie_preds = output[:, :self.num_cie]
            cie_labels = torch.LongTensor(tmp_labels[0]).cuda()
            clus_preds = output[:, self.num_cie: self.num_cie + self.num_clus]
            clus_labels = torch.LongTensor(tmp_labels[1]).cuda() - self.num_cie
            dpt_preds = output[:, -self.num_dpt:]
            dpt_labels = torch.LongTensor(tmp_labels[2]).cuda() - (self.num_cie + self.num_clus)
            loss_cie = torch.nn.functional.cross_entropy(cie_preds, cie_labels)
            loss_clus = torch.nn.functional.cross_entropy(clus_preds, clus_labels)
            loss_dpt = torch.nn.functional.cross_entropy(dpt_preds, dpt_labels)
            loss = loss_cie + loss_clus + loss_dpt
            # loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels.cuda())
        else:
            # the model is specialized
            loss = torch.nn.functional.cross_entropy(output, torch.LongTensor(tmp_labels).view(output.shape[0]).cuda())
            self.log("train_loss_ep", loss, on_step=False, on_epoch=True)
            self.log("train_loss_st", loss, on_step=True, on_epoch=False)
            self.log("train_acc", 100 * accuracy_score(tmp_labels[0],
                                                       torch.argmax(output, dim=-1).detach().cpu().numpy()), on_step=False, on_epoch=True)
#        self.training_losses.append(loss.item())
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        if self.input_type != "userOriented" and self.input_type != "bagTransformer":
            input_tensor = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            output = self.forward(input_tensor)
            labels = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            if self.input_type == "userOriented":
                tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
            if self.input_type == "bagTransformer":
                new_bags = self(bag_matrix.T)
                tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
        if self.type == "poly":
            cie_preds = output[:, :self.num_cie]
            cie_labels = torch.LongTensor(tmp_labels[0]).cuda()
            clus_preds = output[:, self.num_cie: self.num_cie + self.num_clus]
            clus_labels = torch.LongTensor(tmp_labels[1]).cuda() - self.num_cie
            dpt_preds = output[:, -self.num_dpt:]
            dpt_labels = torch.LongTensor(tmp_labels[2]).cuda() - (self.num_cie + self.num_clus)
            val_loss_cie = torch.nn.functional.cross_entropy(cie_preds, cie_labels)
            val_loss_clus = torch.nn.functional.cross_entropy(clus_preds, clus_labels)
            val_loss_dpt = torch.nn.functional.cross_entropy(dpt_preds, dpt_labels)
            val_loss = val_loss_cie + val_loss_clus + val_loss_dpt
        else:
            # the model is specialized
            val_loss = torch.nn.functional.cross_entropy(output,
                                                         torch.LongTensor(tmp_labels).view(output.shape[0]).cuda())
            self.log("val_loss", val_loss, prog_bar=True)
            self.log("valid_acc", 100 * accuracy_score(tmp_labels[0],
                                                       torch.argmax(output, dim=-1).detach().cpu().numpy()))
        return {'val_loss': val_loss}

    # def epoch_end(self):
    #     train_loss_mean = np.mean(self.training_losses)
    #     self.logger.experiment.add_scalar('training_mean_loss', train_loss_mean, global_step=self.current_epoch)
    #     self.training_losses = []

    def configure_optimizers(self):
        if self.hparams.optim == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        else:
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def test_step(self, batch, batch_idx):
        if self.input_type == "userOriented":
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels_one_hot = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            self.test_outputs.append(torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0)).cuda())
        elif self.input_type == "b4Training":
            # ipdb.set_trace()
            input_tensor = self.get_input_tensor(batch)
            self.test_outputs.append(input_tensor)
            tmp_labels = self.get_labels(batch)
            labels_one_hot = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
        elif self.input_type == "bagTransformer":
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels_one_hot = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            new_bags = self(bag_matrix.T)
            tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
            self.test_outputs.append(torch.transpose(tmp, 1, 0))
        else:
            input_tensor = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels_one_hot = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
            self.test_outputs.append(self.forward(input_tensor))
            self.before_training.append(input_tensor)
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
            cie_b4, clus_b4, dpt_b4 = [], [], []
        else:
            cie_preds = outputs[:, :self.num_cie, 0]
            clus_preds = outputs[:, self.num_cie: self.num_cie + self.num_clus, 0]
            dpt_preds = outputs[:, -self.num_dpt:, 0]

        cie_labels = torch.LongTensor([i[0][0] for i in self.test_labels])
        clus_labels = torch.LongTensor([i[1][0] for i in self.test_labels])
        dpt_labels = torch.LongTensor([i[2][0] for i in self.test_labels])

        cie_res = test_for_bag(cie_preds, cie_labels, cie_b4, 0, self.num_cie, "cie")
        clus_res = test_for_bag(clus_preds, clus_labels, clus_b4, self.num_cie, self.num_clus, "clus")
        dpt_res = test_for_bag(dpt_preds, dpt_labels, dpt_b4, self.num_cie + self.num_clus, self.num_dpt, "dpt")

        num_classes = self.num_cie + self.num_clus + self.num_dpt
        general_res = test_for_all_bags(cie_labels, clus_labels, dpt_labels, cie_preds, clus_preds, dpt_preds,
                                        num_classes)
        return {**cie_res, **clus_res, **dpt_res, **general_res}

    def test_spe(self, outputs):
        preds = torch.argsort(outputs.view(-1, self.get_num_classes()), dim=-1, descending=True)
        labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
        res_dict_trained = get_metrics(preds[:, :1].cpu(), labels.cpu(), self.get_num_classes(), self.bag_type, 0)
        for k in [10]:
            tmp = get_metrics_at_k(preds[:, :k].cpu(), labels.cpu(), self.get_num_classes(),
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
                limit = self.num_cie + self.num_clus + self.num_dpt
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
    for preds, labels in tqdm(zip(itertools.chain(cie_preds_at_k, clus_preds_at_k, dpt_preds_at_k), chained_labels),
                              desc="Computing at k=10..."):
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
