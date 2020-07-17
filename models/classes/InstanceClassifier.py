import torch
import ipdb
import pytorch_lightning as pl


class InstanceClassifier(pl.LightningModule):
    def __init__(self, in_size, out_size, input_type):
        super().__init__()
        self.input_type = input_type
        self.lin = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        print("forward")
        ipdb.set_trace()
        return self.lin(x)

    def training_step(self, batch, batch_nb):
        print("training step")
        ipdb.set_trace()

    def validation_step(self, batch, batch_nb):
        print("valid step")
        ipdb.set_trace()

    def test_step(self, batch, batch_nb):
        print("valid step")
        ipdb.set_trace()

    def configure_optimizers(self):
        print("configure optimizer")
        ipdb.set_trace()

        return torch.optim.Adam(self.parameters(), lr=lr)

