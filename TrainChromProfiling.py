import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup


class TrainChromProfiling(pl.LightningModule):
    """Finetune the model or train on the DeepSea features.
    Assume backbone is already fully initialized and checkpoint pre-loaded (if needed)

    TODO:
    - Change the loss function to be more suitable
    """

    def __init__(self, model, sequence_length=256, num_classes=2002, lr=0.0001,
                 weight_decay=0.1, max_epochs=100, limit_train_batches=1000):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = nn.BCELoss() 
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        # define lr scheduler w/ warmup
        num_training_steps = self.limit_train_batches*self.max_epochs
        num_warmup_steps = int(num_training_steps*0.05)
        print('Use get_linear_schedule_with_warmup with num steps ',
              num_training_steps, " and warmup N ", num_warmup_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch
        feats = feats.float().permute(0, 2, 1)

        preds = self.model(feats)
        loss = self.loss_fn(preds, labels.float())
        self.log("Loss/"+mode + "_loss", loss) 
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
