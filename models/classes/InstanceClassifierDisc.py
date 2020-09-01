import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import pickle as pkl
from utils.models import labels_to_one_hot
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class InstanceClassifierDisc(pl.LightningModule):
    def __init__(self, in_size, out_size, hparams, dataset, datadir, desc):
        super().__init__()
        self.lin = torch.nn.Linear(in_size, out_size)

        self.training_losses = []
        self.test_outputs = []
        self.test_labels_one_hot = []
        self.test_labels = []
        self.test_ppl_id = []
        self.type = desc.split("_")[1]
        if self.type == 'spe':
            self.bag_type = desc.split("_")[2]
            torch.nn.init.eye_(self.lin.weight)
            torch.nn.init.zeros_(self.lin.bias)

        self.input_type = hparams.input_type
        self.num_cie = dataset.num_cie
        self.num_clus = dataset.num_clus
        self.num_dpt = dataset.num_dpt

        self.hparams = hparams
        self.data_dir = datadir
        self.description = desc

        ### debug
        self.before_training = []

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_nb):
        if self.input_type != "userOriented":
            input_tensor = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            output = self.forward(input_tensor)
            labels = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
            output = torch.transpose(tmp, 1, 0)
        if self.type == "poly":
            loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels.cuda())
        else:
            # the model is specialized
            loss = torch.nn.functional.cross_entropy(output, torch.LongTensor(tmp_labels).view(output.shape[0]).cuda())
        self.training_losses.append(loss.item())

        preds = [i.item() for i in torch.argmax(output, dim=1)]
        res_dict = get_metrics(preds, tmp_labels[0], self.get_num_classes(), "train")
        tensorboard_logs = {**res_dict, 'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        if self.input_type != "userOriented":
            input_tensor = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            output = self.forward(input_tensor)
            labels = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
            output = torch.transpose(tmp, 1, 0)
        if self.type == "poly":
            val_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels.cuda())
        else:
            # the model is specialized
            val_loss = torch.nn.functional.cross_entropy(output,
                                                         torch.LongTensor(tmp_labels).view(output.shape[0]).cuda())
        tensorboard_logs = {'val_loss': val_loss}
        return {'loss': val_loss, 'log': tensorboard_logs}

    def epoch_end(self):
        train_loss_mean = np.mean(self.training_losses)
        self.logger.experiment.add_scalar('training_mean_loss', train_loss_mean, global_step=self.current_epoch)
        self.training_losses = []

    def validation_end(self, outputs):
        return outputs[-1]

    def configure_optimizers(self):
        wd = 0
        if self.type == "spe":
            if self.bag_type == "clus":
                wd = .8
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=wd)

    def test_step(self, batch, batch_idx):
        if self.input_type != "userOriented":
            input_tensor = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels_one_hot = labels_to_one_hot(input_tensor.shape[0], tmp_labels, self.get_num_classes())
            self.test_outputs.append(self.forward(input_tensor))
            self.before_training.append(input_tensor)
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            tmp_labels = self.get_labels(batch)
            labels_one_hot = labels_to_one_hot(profiles.shape[0], tmp_labels, self.get_num_classes())
            self.test_outputs.append(torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0)).cuda())
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
            cie_b4 = torch.stack(self.before_training)[:, 0, :self.num_cie]
            clus_b4 = torch.stack(self.before_training)[:, 0, self.num_cie: self.num_cie + self.num_clus]
            dpt_b4 = torch.stack(self.before_training)[:, 0, -self.num_dpt:]
        else:
            cie_preds = outputs[:, :self.num_cie, 0]
            clus_preds = outputs[:, self.num_cie: self.num_cie + self.num_clus, 0]
            dpt_preds = outputs[:, -self.num_dpt:, 0]

        cie_labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
        clus_labels = torch.LongTensor([i[1][0] for i in self.test_labels]).cuda()
        dpt_labels = torch.LongTensor([i[2][0] for i in self.test_labels]).cuda()

        ipdb.set_trace()
        cie_res = test_for_bag(cie_preds, cie_labels, cie_b4, 0, self.num_cie, "cie")
        clus_res = test_for_bag(clus_preds, clus_labels, clus_b4, self.num_cie, self.num_clus, "clus")
        dpt_res = test_for_bag(dpt_preds, dpt_labels, dpt_b4, self.num_cie + self.num_clus, self.num_dpt, "dpt")
        return {**cie_res, **clus_res, **dpt_res}

    def test_spe(self, outputs):
        preds = torch.argmax(outputs.view(-1, self.get_num_classes()), dim=1)
        labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
        b4_training = torch.argmax(torch.stack(self.before_training)[:, 0, :], dim=1)
        res_dict_trained = get_metrics(preds.cpu(), labels.cpu(), self.get_num_classes(), self.bag_type + "_trained")
        res_dict_b4_training = get_metrics(b4_training.cpu(), labels.cpu(), self.get_num_classes(), self.bag_type + "_b4")

        return {**res_dict_b4_training, **res_dict_trained}


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


def test_for_bag(preds, labels, b4_training, offset, num_classes, bag_type):
    tmp = [i.item() for i in torch.argmax(preds, dim=1)]
    b4_train = [i.item() + offset for i in torch.argmax(b4_training, dim=1)]
    predicted_classes = [i + offset for i in tmp]
    res_dict_trained = get_metrics(predicted_classes, labels.cpu(), num_classes, bag_type + "_trained")
    res_dict_b4_training = get_metrics(b4_train, labels.cpu(), num_classes, bag_type + "_b4")
    return {**res_dict_b4_training, **res_dict_trained}


def get_average_metrics(res_dict):
    precision = []
    recall = []
    numerical_keys = [i for i in res_dict.keys()][:-3]
    for k in numerical_keys:
        precision.append(res_dict[k]["precision"])
        recall.append(res_dict[k]["recall"])
    return np.mean(precision), np.mean(recall)


def get_metrics(preds, labels, num_classes, handle):
    res_dict = {
        "acc_" + handle: accuracy_score(preds, labels) * 100,
        "precision_" + handle: precision_score(preds, labels, average='weighted',
                                               labels=range(num_classes)) * 100,
        "recall_" + handle: recall_score(preds, labels, average='weighted', labels=range(num_classes)) * 100,
        "f1_" + handle: f1_score(preds, labels, average='weighted', labels=range(num_classes)) * 100}
    return res_dict
