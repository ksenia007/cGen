import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from transformers import get_linear_schedule_with_warmup


class SimCGen(pl.LightningModule):
    def __init__(self, lr, temperature, weight_decay, backbone,
                 max_epochs=100, limit_train_batches=1000):
        """Note that the backbone is assumed to have the top MLP layer already"""
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.backbone = backbone
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.automatic_optimization = True

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        num_training_steps = self.limit_train_batches*self.max_epochs
        num_warmup_steps = int(num_training_steps*0.1)
        print('Use get_linear_schedule_with_warmup with num steps ',
              num_training_steps, " and warmup N ", num_warmup_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch=-1)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def predict(self, batch):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.float().permute(0, 1, 3, 2)
        imgs = torch.squeeze(imgs)
        feats, _ = self.backbone(imgs)
        feats = feats.view(feats.size(0), -1)
        return feats

    def info_nce_loss(self, preds, mode="train"):
        cos_sim = F.cosine_similarity(
            preds[:, None, :], preds[None, :, :], dim=-1)
        self_mask = torch.eye(
            cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        self.log(mode + "_sim_pos", -cos_sim[pos_mask].mean(), sync_dist=True)
        self.log(mode + "_logsumexp", torch.logsumexp(cos_sim, dim=-1).mean(), sync_dist=True)
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        self.log("Loss/" + mode + "_loss", nll, sync_dist=True)

        return nll

    def training_step(self, batch, batch_idx):
        pred = self.predict(batch)
        return {"loss": self.info_nce_loss(pred, mode="train"), "pred": pred}

    def validation_step(self, batch, batch_idx):
        pred = self.predict(batch)
        return {"loss": self.info_nce_loss(pred, mode="val"), "pred": pred}


    def step_end(self, step_outputs):
        if type(step_outputs) == list:
            pred_views = None
            for out in step_outputs:
                pred = out['pred']
                if pred_views is None:
                    pred_views = [[]] * len(pred)
                for ix, view in enumerate(pred):
                    pred_views[ix].append(view)

            loss = self.info_nce_loss([torch.cat(p, dim=0) for p in pred_views])
            return loss
        else:
            return step_outputs['loss']


    def training_step_end(self, training_step_outputs):
        return self.step_end(training_step_outputs)

    def validation_step_end(self, validation_step_outputs):
        return self.step_end(validation_step_outputs)
