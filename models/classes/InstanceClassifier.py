import ipdb
import torch
import os
import pytorch_lightning as pl
import numpy as np
import pickle as pkl
from utils.models import labels_to_one_hot
from sklearn.metrics import confusion_matrix


class InstanceClassifier(pl.LightningModule):
    def __init__(self, in_size, out_size, hparams, dataset, datadir, desc):
        super().__init__()
        self.lin = torch.nn.Linear(in_size, out_size)

        self.training_losses = []
        self.test_outputs = []
        self.test_labels_one_hot = []
        self.test_labels = []
        self.test_ppl_id = []

        self.input_type = hparams.input_type
        self.lr = hparams.lr
        self.num_cie = dataset.num_cie
        self.num_clus = dataset.num_clus
        self.num_dpt = dataset.num_dpt

        self.data_dir = datadir
        self.description = desc

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_nb):
        if self.input_type == "matMul":
            input_tensor = torch.matmul(batch[-1], torch.transpose(batch[1], 2, 1)).squeeze(-1)
            labels = labels_to_one_hot(input_tensor.shape[0], [batch[2], batch[3], batch[4]], input_tensor.shape[-1])
            assert torch.sum(labels) == 3 * len(input_tensor)
            output = self.forward(input_tensor)
            loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels.cuda())
        else:
            raise Exception("Wrong input data specified: " + str(self.input_type))
        tensorboard_logs = {'train_loss': loss}
        self.training_losses.append(loss.item())
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        if self.input_type == "matMul":
            input_tensor = torch.matmul(batch[-1], torch.transpose(batch[1], 2, 1)).squeeze(-1)
            labels = labels_to_one_hot(input_tensor.shape[0], [batch[2], batch[3], batch[4]], input_tensor.shape[-1])
            assert torch.sum(labels) == 3 * len(input_tensor)
            output = self.forward(input_tensor)
            val_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), labels.cuda())
        else:
            raise Exception("Wrong input data specified: " + str(self.input_type))
        tensorboard_logs = {'val_loss': val_loss}
        return {'loss': val_loss, 'log': tensorboard_logs}

    def epoch_end(self):
        train_loss_mean = np.mean(self.training_losses)
        self.logger.experiment.add_scalar('training_mean_loss', train_loss_mean, global_step=self.current_epoch)
        self.training_losses = []

    def validation_end(self, outputs):
        return outputs[-1]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def test_step(self, batch, batch_idx):
        if self.input_type == "matMul":
            input_tensor = torch.matmul(batch[-1], torch.transpose(batch[1], 2, 1)).squeeze(-1)
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

        # slicing labels per class type
        cie_labels = torch.LongTensor([i[0][0] for i in self.test_labels]).cuda()
        clus_labels = torch.LongTensor([i[1][0] for i in self.test_labels]).cuda()
        dpt_labels = torch.LongTensor([i[2][0] for i in self.test_labels]).cuda()

        ci_preds, ci_acc, ci_cm = test_for_bag(cie_preds, cie_labels, "cie")
        cl_preds, cl_acc, cl_cm = test_for_bag(clus_preds, clus_labels, "clus")
        d_preds, d_acc, d_cm = test_for_bag(dpt_preds, dpt_labels, "dpt")

        self.save_outputs(ci_preds, cie_labels, ci_cm,
                          cl_preds, clus_labels, cl_cm,
                          d_preds, dpt_labels, d_cm)
        # # making a tensor out of the one hot labels
        # labels_one_hot = torch.stack(self.test_labels_one_hot).cuda()
        # cie_labels_one_hot = labels_one_hot[:, 0, :self.num_cie]
        # clus_labels_one_hot = labels_one_hot[:, 0, self.num_cie: self.num_cie + self.num_clus]
        # dpt_labels_one_hot = labels_one_hot[:, 0, -self.num_dpt:]

        return {"cie_acc": ci_acc,
                "clus_acc": cl_acc,
                "dpt_acc": d_acc}

    def save_outputs(self, ci_preds, cie_labels, ci_cm,
                     cl_preds, clus_labels, cl_cm,
                     d_preds, dpt_labels, d_cm):
        res = {"cie": {"preds": ci_preds,
                       "labels": cie_labels,
                       "cm": ci_cm},
               "clus": {"preds": cl_preds,
                        "labels": clus_labels,
                        "cm": cl_cm},
               "dpt": {"preds": d_preds,
                       "labels": dpt_labels,
                       "cm": d_cm},
               "ppl": self.test_ppl_id
               }
        tgt_file = os.path.join(self.data_dir, "OUTPUTS_" + self.description + ".pkl")
        with open(tgt_file, 'wb') as f:
            pkl.dump(res, f)


def test_for_bag(pred, labels, bag_type):
    predicted_classes = [i.item() for i in torch.argmax(pred, dim=1)]
    good_pred = 0
    wrong_pred = 0
    for pred, label in zip(predicted_classes, labels):
        if pred == label.item():
            good_pred += 1
        else:
            wrong_pred += 1
    global_accuracy = 100 * good_pred / (good_pred + wrong_pred)
    cm = confusion_matrix(labels.cpu().numpy(), np.asarray(predicted_classes))
    print("Overall accuracy for " + bag_type + " is: " + str(global_accuracy) + " %")

    return predicted_classes, global_accuracy, cm


def test_for_all_bags(pred, truth, total_ppl):
    tp = torch.sum(torch.sum(pred & truth, dim=1))
    fp = torch.sum(torch.sum((truth == 0) & (pred == 1), dim=1))
    fn = torch.sum(torch.sum((truth == 1) & (pred == 0), dim=1))
    tn = torch.sum(torch.sum((truth == 0) & (pred == 0), dim=1))

    neg_ratio = (tn.item() + fn.item()) / total_ppl
    pos_ratio = (tp.item() + fp.item()) / total_ppl

    precision = tp.type(torch.float32) / ((tp + fp).type(torch.float32) + 1e-15)
    recall = tp.type(torch.float32) / (tp + fn).type(torch.float32)
    acc = (tn.item() + tp.item()) / total_ppl
    if (recall + precision) != 0:
        f1 = 2 * (precision * recall) / (recall + precision)
    else:
        f1 = torch.FloatTensor([0.0])
    metrics = {"acc": acc,
               "precision": precision.item(),
               "recall": recall.item(),
               "F1": f1.item(),
               "pos_ratio": pos_ratio,
               "neg_ratio": neg_ratio}
    return metrics
