import torch
import pytorch_lightning as pl


class InstanceClassifier(pl.LightningModule):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.lin = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_nb):
        pass

    def configure_optimizers(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

