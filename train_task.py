import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def expr_loss(input, target, noise=False):
    """Loss w/ the noise"""

    loss = nn.MSELoss()
    if noise:
        # add noise
        div_v = 100
        print('adding noise, shifted, div', div_v)
        noise_vector = torch.sub(torch.rand(target.shape), 0.5) # shift from (0,1) to (-0.5, 0.5) range
        noise_vector = torch.div(noise_vector, div_v).cuda()
        target = torch.add(target, noise_vector)

    if input.shape != target.shape:
        print("ERROR in sizes:", input.shape, target.shape)
        raise ValueError


    return loss(input, target)



class TrainChromProfiling(pl.LightningModule):
    """Finetune the model or train on the DeepSea features.
    Assume backbone is already fully initialized and checkpoint pre-loaded (if needed)

    TODO:
    - Change the loss function to be more suitable
    """

    def __init__(self, model, sequence_length=256, num_classes=2002, lr=0.0001,
                 weight_decay=0.1, max_epochs=100, limit_train_batches=1000, 
                 optim_name=''):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = nn.BCELoss() # nn.BCEWithLogitsLoss()
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay, momentum=0.9)
        return optimizer


    def _calculate_loss(self, batch, mode="train"):
        output = batch
        feats, labels, additional = None, None, None
        if len(output) > 2:
            feats, labels, additional = batch
        else:
            feats, labels = batch
        feats = feats.float().permute(0, 2, 1)

        preds = self.model(feats)
        loss = self.loss_fn(preds, labels.float())
        self.log("Loss/"+mode + "_loss", loss) #, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")



class TrainExpression(pl.LightningModule):
    """Finetune the model or train on gene expression as a regression task.
    Assume backbone is already fully initialized and checkpoint pre-loaded (if needed)
    """

    def __init__(self, model, sequence_length=256, num_classes=54, lr=0.0001,
                 weight_decay=0.1, max_epochs=100, limit_train_batches=1000, 
                 optim_name=''):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optim_name = optim_name
        self.loss_fn = expr_loss #nn.MSELoss()
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches

    def configure_optimizers(self):
        if self.optim_name=='SGD':
            return optim.SGD(self.parameters(), lr=self.hparams.lr, 
                             momentum=0.9)
        else:
            return optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _calculate_loss(self, batch, mode="train"):
        output = batch
        feats, labels, additional = None, None, None
        if len(output) > 2:
            feats, labels, additional = batch
        else:
            feats, labels = batch

        feats = feats.float().permute(0, 2, 1)

        preds = self.model(feats, additional)

        loss = self.loss_fn(preds, labels.float())
        self.log("Loss/"+mode + "_loss", loss) #, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

