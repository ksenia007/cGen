import pytorch_lightning as pl


class ResetSampler(pl.Callback):
    def _prepare_epoch(self, trainer, model, epoch):
        if trainer.train_dataloader is None:
            return None
        try: trainer.train_dataloader.dataset.datasets
        except:
            return None
        if not hasattr(trainer.train_dataloader.dataset.datasets, 'sampler'):
            return None
        sampler = trainer.train_dataloader.dataset.datasets.sampler
        if 'train' in sampler.cache_modes:
            sampler.seed = sampler.seed + epoch
            sampler._reset_train = True

    def on_train_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)


