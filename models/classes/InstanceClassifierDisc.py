import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import pickle as pkl
from utils.models import labels_to_one_hot
from sklearn.metrics import confusion_matrix, classification_report


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

        self.input_type = hparams.input_type
        self.num_cie = dataset.num_cie
        self.num_clus = dataset.num_clus
        self.num_dpt = dataset.num_dpt

        self.hparams = hparams
        self.data_dir = datadir
        self.description = desc

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_nb):
        input_tensor = self.get_input_tensor(batch)
        tmp_labels = self.get_labels(batch)
        labels = labels_to_one_hot(input_tensor.shape[0], tmp_labels, input_tensor.shape[-1])
        output = self.forward(input_tensor)
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels.cuda())
        tensorboard_logs = {'train_loss': loss}
        self.training_losses.append(loss.item())
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_tensor = self.get_input_tensor(batch)
        tmp_labels = self.get_labels(batch)
        labels = labels_to_one_hot(input_tensor.shape[0], tmp_labels, input_tensor.shape[-1])
        output = self.forward(input_tensor)
        val_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels.cuda())
        tensorboard_logs = {'val_loss': val_loss}
        return {'loss': val_loss, 'log': tensorboard_logs}

    def epoch_end(self):
        train_loss_mean = np.mean(self.training_losses)
        self.logger.experiment.add_scalar('training_mean_loss', train_loss_mean, global_step=self.current_epoch)
        self.training_losses = []

    def validation_end(self, outputs):
        return outputs[-1]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def test_step(self, batch, batch_idx):
        if self.input_type == "matMul":
            if len(batch[1].shape) > 2:
                ppl_tensor = torch.transpose(batch[1], 2, 1)
            else:
                ppl_tensor = torch.transpose(batch[1], 1, 0)
            tmp = torch.matmul(batch[-1], ppl_tensor).squeeze(-1)
            input_tensor = tmp.view(len(batch[0]), -1)
            labels = [batch[2], batch[3], batch[4]]
            labels_one_hot = labels_to_one_hot(input_tensor.shape[0], [batch[2], batch[3], batch[4]],
                                               input_tensor.shape[-1])
            assert torch.sum(labels_one_hot) == 3 * len(input_tensor)
            self.test_outputs.append(self.forward(input_tensor))
            self.test_labels_one_hot.append(labels_one_hot)
            self.test_labels.append(labels)
            self.test_ppl_id.append(batch[0])

    def test_epoch_end(self, outputs):
        outputs = torch.stack(self.test_outputs)
        # slicing predictions per class type
        cie_preds = outputs[:, 0, :self.num_cie]
        clus_preds = outputs[:, 0, self.num_cie: self.num_cie + self.num_clus]
        dpt_preds = outputs[:, 0, -self.num_dpt:]

        cie_labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
        clus_labels = torch.LongTensor([i[1][0] for i in self.test_labels]).cuda()
        dpt_labels = torch.LongTensor([i[2][0] for i in self.test_labels]).cuda()

        ci_preds, ci_cm, ci_res = test_for_bag(cie_preds, cie_labels, offset=0)
        cl_preds, cl_cm, cl_res = test_for_bag(clus_preds, clus_labels, offset=self.num_cie)
        d_preds, d_cm, d_res = test_for_bag(dpt_preds, dpt_labels, offset=self.num_cie + self.num_clus)
        self.save_outputs(ci_preds, cie_labels, ci_cm, ci_res,
                          cl_preds, clus_labels, cl_cm, cl_res,
                          d_preds, dpt_labels, d_cm, d_res)

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
            tmp_labels = [batch[2]]
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
            ipdb.set_trace()
        else:
            raise Exception("Wrong input data specified: " + str(self.input_type))
        return input_tensor


def test_for_bag(preds, labels, offset):
    tmp = [i.item() for i in torch.argmax(preds, dim=1)]
    predicted_classes = [i + offset for i in tmp]
    cm = confusion_matrix(labels.cpu().numpy(), np.asarray(predicted_classes))
    results = classification_report(predicted_classes, labels.cpu(), output_dict=True)
    return predicted_classes, cm, results


def get_average_metrics(res_dict):
    precision = []
    recall = []
    numerical_keys = [i for i in res_dict.keys()][:-3]
    for k in numerical_keys:
        precision.append(res_dict[k]["precision"])
        recall.append(res_dict[k]["recall"])
    return np.mean(precision), np.mean(recall)
