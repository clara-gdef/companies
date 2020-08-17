import ipdb
import torch
import pytorch_lightning as pl
import numpy as np
from utils.models import labels_to_one_hot, class_to_one_hot


class InstanceClassifier(pl.LightningModule):
    def __init__(self, in_size, out_size, hparams, dataset):
        super().__init__()
        self.lin = torch.nn.Linear(in_size, out_size)

        self.training_losses = []
        self.test_outputs = []
        self.test_labels = []

        self.input_type = hparams.input_type
        self.lr = hparams.lr
        self.num_cie = dataset.num_cie
        self.num_clus = dataset.num_clus
        self.num_dpt = dataset.num_dpt

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
            labels = labels_to_one_hot(input_tensor.shape[0], [batch[2], batch[3], batch[4]], input_tensor.shape[-1])
            assert torch.sum(labels) == 3 * len(input_tensor)
            self.test_outputs.append(self.forward(input_tensor))
            self.test_labels.append(labels)

    def test_epoch_end(self, outputs):
        num_ppl = len(outputs)
        outputs = torch.stack(self.test_outputs)
        # slicing predictions per class type
        cie_preds = outputs[:, 0, :self.num_cie]
        clus_preds = outputs[:, 0, self.num_cie: self.num_cie + self.num_clus]
        dpt_preds = outputs[:, 0, -self.num_dpt:]
        # making a tensor out of the labels
        labels = torch.stack(self.test_labels).cuda()
        # slicing labels per class type
        cie_labels = labels[:, 0, :self.num_cie]
        clus_labels = labels[:, 0, self.num_cie: self.num_cie + self.num_clus]
        dpt_labels = labels[:, 0, -self.num_dpt:]
        test_for_cie(cie_preds, cie_labels, num_ppl)
        ipdb.set_trace()


def test_for_cie(pred, labels, total_ppl):
    predicted_classes = [i.item() for i in torch.argmax(pred, dim=1)]
    pred_one_hot = class_to_one_hot(len(pred), predicted_classes, pred.shape[-1])
    pred = pred_one_hot.type(torch.cuda.ByteTensor)
    truth = labels.type(torch.cuda.ByteTensor)

    tp = torch.sum(torch.sum(pred & truth, dim=1))
    fp = torch.sum(torch.sum((truth == 0) & (pred == 1), dim=1))
    fn = torch.sum(torch.sum((truth == 1) & (pred == 0), dim=1))
    tn = torch.sum(torch.sum((truth == 0) & (pred == 0), dim=1))

    assert tp + fp + fn + tn == total_ppl

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


def test_for_clus():
    pass


def test_for_dpt():
    pass


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