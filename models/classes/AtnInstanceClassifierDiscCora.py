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


class AtnInstanceClassifierDiscCora(pl.LightningModule):
    def __init__(self, in_size, out_size, hparams, input_type, ft_type, num_tracks, datadir, desc, frozen, init_type, optim):
        super().__init__()
        self.input_type = hparams.input_type
        self.num_tracks = num_tracks
        self.ft_type = ft_type
        self.input_type = input_type
        self.init_type = init_type
        self.optim = optim

        self.hp = hparams
        self.data_dir = datadir
        self.description = desc

        with open(os.path.join(self.data_dir, "cora_area_fqc.pkl"), 'rb') as f:
            tmp = pkl.load(f)
        self.class_weights = tmp.cuda()

        #self.atn_layer = torch.nn.Linear(300, 1)
        self.atn_layer = torch.nn.Sequential(torch.nn.Linear(300, 150, bias=False),
                                             torch.nn.Linear(150, 1, bias=False))
        ############
        # torch.nn.init.ones_(self.atn_layer.weight)
        # torch.nn.init.constant_(self.atn_layer.weight, 1/(12*300))
        # torch.nn.init.zeros_(self.atn_layer.bias)
        ###########
        if self.input_type == "hadamard":
            self.lin_dim_reduction = torch.nn.Linear(in_size, self.hp.middle_size)
            self.lin_class_prediction = torch.nn.Linear(self.hp.middle_size, out_size)
            torch.nn.init.eye_(self.lin_dim_reduction.weight)
            torch.nn.init.eye_(self.lin_class_prediction.weight)
            torch.nn.init.zeros_(self.lin_dim_reduction.bias)
            torch.nn.init.zeros_(self.lin_class_prediction.bias)
        else:
            self.lin = torch.nn.Linear(in_size, out_size)
            if init_type == "eye":
                torch.nn.init.eye_(self.lin.weight)
                torch.nn.init.zeros_(self.lin.bias)
            elif init_type == "zeros":
                torch.nn.init.zeros_(self.lin.weight)
                torch.nn.init.zeros_(self.lin.bias)
        if frozen:
            self.lin.requires_grad = False

        if self.hp.log_cm == "True":
            self.train_outputs = []
            self.valid_outputs = []
            self.train_labels = []
            self.valid_labels = []

        self.training_losses = []
        self.test_outputs = []
        self.test_labels_one_hot = []
        self.test_labels = []
        self.test_ppl_id = []

        ### debug
        self.before_training = []

    def forward(self, tmp_people, bags):
        people = torch.stack(tmp_people).type(torch.FloatTensor).cuda()
        atn = torch.relu(self.atn_layer(people))
        normed_atn = atn.clone()
        for ind, sample in enumerate(atn):
            normed_atn[ind] = torch.softmax(atn[ind], dim=0)
        new_people = self.ponderate_jobs(people, normed_atn)
        if self.input_type == "bagTransformer":
            mat = torch.diag(self.lin.weight).unsqueeze(1)
            out = bags * mat + self.lin.bias.view(bags.shape[0], -1)
            return out.T, new_people
        elif self.input_type == "hadamard":
            affinities = torch.matmul(new_people.cuda(), bags.cuda())
            transformed_input = torch.relu(self.lin_dim_reduction(affinities))
            return self.lin_class_prediction(transformed_input)
        else:
            affinities = torch.matmul(new_people.cuda(), bags.cuda())
            return self.lin(affinities)

    def training_step(self, batch, batch_nb):
        labels = batch[-2]
        if self.input_type != "userOriented" and self.input_type != "bagTransformer":
            bags, profiles = self.get_input_tensor(batch)
            output = self.forward(profiles, bags)
            if self.hp.log_cm == "True":
                self.train_outputs.append(output)
                self.train_labels.append(labels)
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            if self.input_type == "userOriented":
                tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
            if self.input_type == "bagTransformer":
                new_bags = self(bag_matrix.T)
                tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
        if self.hp.weight_classes == "True":
            loss = torch.nn.functional.cross_entropy(output, labels, weight=self.class_weights)
        else:
            loss = torch.nn.functional.cross_entropy(output, labels)
        self.log("train_loss_CE", loss)
        self.training_losses.append(loss.item())
        self.log("train_acc", 100*accuracy_score(labels.cpu().numpy(), torch.argmax(output, dim=-1).detach().cpu().numpy()))
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        labels = batch[-2]
        if self.input_type != "userOriented" and self.input_type != "bagTransformer":
            bags, profiles = self.get_input_tensor(batch)
            output = self.forward(profiles, bags)
            if self.hp.log_cm == "True":
                self.valid_outputs.append(output)
                self.valid_labels.append(labels)
        else:
            bag_matrix, profiles = self.get_input_tensor(batch)
            if self.input_type == "userOriented":
                tmp = torch.matmul(self.forward(bag_matrix), torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
            if self.input_type == "bagTransformer":
                new_bags = self(bag_matrix.T)
                tmp = torch.matmul(new_bags, torch.transpose(profiles, 1, 0))
                output = torch.transpose(tmp, 1, 0)
        if self.hp.weight_classes == "True":
            val_loss = torch.nn.functional.cross_entropy(output, labels, weight=self.class_weights)
        else:
            val_loss = torch.nn.functional.cross_entropy(output, labels)
        self.log("val_loss_CE", val_loss)
        self.log("valid_acc", 100*accuracy_score(labels.cpu().numpy(), torch.argmax(output, dim=-1).detach().cpu().numpy()))
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.optim == "adam":
            return torch.optim.Adam(params, lr=self.hp.lr, weight_decay=self.hp.wd)
        else:
            return torch.optim.SGD(params, lr=self.hp.lr, weight_decay=self.hp.wd)


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
            bags, profiles = self.get_input_tensor(batch)
            output = self.forward(profiles, bags)
            labels = batch[-2]
            self.test_outputs.append(self.forward(profiles, bags))
            # self.before_training.append(input_tensor)
        self.test_labels.append(labels)
        self.test_ppl_id.append(batch[0])

    def test_epoch_end(self, outputs):
        outputs = torch.stack(self.test_outputs)
        return self.test_spe(outputs)

    def test_spe(self, outputs):
        preds = torch.argsort(outputs.view(-1, self.num_tracks), dim=-1, descending=True)
        labels = torch.LongTensor([i.item() for i in self.test_labels]).cuda()
        res_dict_trained = get_metrics(preds[:, :1].cpu(), labels.cpu(), self.num_tracks, "tracks", 0)
        for k in [3]:
            tmp = get_metrics_at_k(preds[:, :k].cpu(), labels.cpu(), self.num_tracks, "tracks_@" + str(k), 0)
            res_dict_trained = {**res_dict_trained, **tmp}
        # self.save_bag_outputs(preds, labels, confusion_matrix(preds[:, :1].cpu(), labels.cpu()), res_dict_trained)
        cm = confusion_matrix(labels.cpu().numpy(), preds[:, 0].cpu().numpy())
        ipdb.set_trace()
        with open(os.path.join(self.data_dir, "cm_" + self.description + ".pkl"), "wb") as f:
            pkl.dump(cm, f)
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

    def get_confusion_matrix(self, preds, labels):
        sorted_preds = torch.argmax(preds, dim=-1).view(-1, 1)
        return  confusion_matrix(labels.view(-1, 1).squeeze(-1).detach().cpu().numpy(),
                                 sorted_preds.squeeze(-1).detach().cpu().numpy())

    def log_confusion_matrix(self, preds, labels, handle):
        cm = self.get_confusion_matrix(preds, labels)
        tgt_file = os.path.join(self.data_dir, "cm_" + handle + "_" + self.description + ".pkl")
        with open(tgt_file, "wb") as f:
            pkl.dump(cm, f)

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
        profiles = batch[2]
        bags = batch[-1].T
        return bags, profiles

    def get_outputs_and_labels(self, test_loader):
        idx = []
        for batch in tqdm(test_loader, desc="Testing..."):
            idx.append(batch[0][0])
            self.test_step(batch, 0)
        outputs = torch.stack(self.test_outputs)
        preds = outputs[:, 0, :]
        labels = torch.LongTensor([i.item() for i in self.test_labels])
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

    def ponderate_jobs(self, people, atn):
        new_people = torch.zeros(len(people), 300).cuda()
        for num, person in enumerate(people):
            job_counter = 0
            new_p = torch.zeros(300).cuda()
            for j, job in enumerate(person):
                # that means the job is a placeholder, and equal to zero everywhere
                if (job != torch.zeros(300).cuda()).all():
                    job_counter += 1
                    new_p += atn[num][j] * job
            new_people[num] = new_p / job_counter
        return new_people


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
