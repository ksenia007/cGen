""" Code to train the model.

TODO:
- put a main+argument parser to pass some params from .sh (such as transforms, batch size, lr and other)
- check cuda() training
- add logger instead of print()
"""

import json
import logging
import os
import uuid
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

import augmentations as aug
from custom_callbacks import ResetSampler
import dataloader as dl_module
import model as model_module
import train_task

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train(**kwargs):
    """
    TODO: change validation instead of test data (?)
    """
    # define dataset and augmentations
    if 'dataloader' not in kwargs:
        logging.info("Warning: No dataloader specified. Default to 'GenomicData'")
        kwargs['dataloader'] = 'GenomicData'

    dl_args = {'seq_length': kwargs['seq_length'],
              'train_batch_size': kwargs['train_batch_size'],
              'eval_batch_size': kwargs['eval_batch_size'],
              'data_dir': kwargs['data_dir']}

    if 'h5_filepath' in kwargs:
        dl_args['h5_filepath'] = kwargs['h5_filepath']

    if 'use_additional' in kwargs:
        dl_args['use_additional'] = kwargs['use_additional']

    if 'n_cpus' in kwargs:
        dl_args['n_cpus'] = kwargs['n_cpus']

    if 'targets_file' in kwargs:
        dl_args['targets_file'] = kwargs['targets_file']
    if 'txt_file' in kwargs:
        dl_args['txt_file'] = kwargs['txt_file']

    dl_class = getattr(dl_module, kwargs['dataloader'])
    dataloader = dl_class('finetune', **dl_args)
    dataloader.prepare_data()
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()
    logging.info("Prepared data!")

    # checkpoint
    checkpoint_last = ModelCheckpoint(
         save_top_k=1,
         mode='max', monitor='epoch',
         filename="model-"+kwargs['jobID']+"-epoch={epoch:02d}-val_loss={Loss/val_loss:.3f}",
         dirpath=kwargs['checkpoint_folder'],
         auto_insert_metric_name=False)

    checkpoint_lowest = ModelCheckpoint(
        mode='min',
        monitor='Loss/val_loss',
        filename="model-"+kwargs['jobID']+"-epoch={epoch:02d}-val_loss={Loss/val_loss:.3f}",
        dirpath=kwargs['checkpoint_folder'],
        auto_insert_metric_name=False)

    # lr checking
    lr_monitor = LearningRateMonitor(logging_interval='step')


    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 2002
    # define train
    if 'task' not in kwargs:
        kwargs['task'] = 'TrainChromProfiling'

    task = getattr(train_task, kwargs['task'])
    finetune_cls = getattr(model_module, 'Model{0}'.format(kwargs['task'].split('Train')[1]))
    model = finetune_cls(model_type=kwargs["model_type"],
                         model_name=kwargs['model_name'],
                         sequence_length=kwargs['seq_length'],
                         num_classes=kwargs['num_classes'],
                         checkpoint=kwargs['checkpoint'],
                         hidden_size = kwargs['hidden_size'])


    trainer_args = kwargs['trainer']
    if 'accelerator' not in trainer_args:
        trainer_args['accelerator'] = 'auto'

    model = task(model, lr=kwargs['lr'],
                 weight_decay=kwargs['weight_decay'],
                 limit_train_batches=trainer_args['limit_train_batches'],
                 max_epochs=trainer_args['max_epochs'])

    logger = TensorBoardLogger(kwargs['tensorboard_folder'], name="")

    trainer = pl.Trainer(gradient_clip_val=0.5,
                         logger=logger,
                         callbacks=[checkpoint_last,
                                    checkpoint_lowest,
                                    ResetSampler(), lr_monitor],
                         **trainer_args)
    logging.info("Set up training, starting to train")

    # train
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setup", help='A required .yaml file with the training setup')
    args = parser.parse_args()

    with open(args.setup) as f:
        setup_args = yaml.safe_load(f)

    # add job ID
    id_job = str(uuid.uuid4().fields[0])
    print("*Starting job: ", id_job)
    setup_args['jobID'] = id_job

     # load the checkpoints using the ID
    if 'checkpoint_ID' in setup_args:
        print('Checkpoint ID is specified')
        location_checkpoint = os.path.join(setup_args['checkpoint_folder_load'], setup_args['checkpoint_ID'], 'checkpoints')
        files = os.listdir(location_checkpoint)
        files = [i for i in files if '.ckpt' in i]
        print('Available checkpoints', files)

    output_folder_BASE = setup_args['output_folder']

    for f_use in files:
        # epoch
        epoch_name = f_use.split('-')[2]
        print('Epoch', epoch_name)
        # create a nested folder structure
        setup_args['output_folder'] = os.path.join(
                output_folder_BASE, id_job, epoch_name)

        setup_args['tensorboard_folder'] = os.path.join(
            setup_args['output_folder'], "tensorboard_logs")
        setup_args['checkpoint_folder'] = os.path.join(
            setup_args['output_folder'], "checkpoints")

        os.makedirs(setup_args['output_folder'], exist_ok=True)
        os.makedirs(setup_args['tensorboard_folder'], exist_ok=True)
        os.makedirs(setup_args['checkpoint_folder'], exist_ok=True)

        # save setup used in the same folder for reproducibility
        with open(setup_args['output_folder']+'/setup.json', 'w', encoding='utf-8') as f:
            json.dump(setup_args, f, ensure_ascii=False, indent=4)

        # for files 
        setup_args['checkpoint'] = os.path.join(location_checkpoint, f_use)
        print('Using checkpoint', setup_args['checkpoint'])

        # start logging session
        logging.basicConfig(
            filename=setup_args['output_folder']+'/log.log', level=logging.DEBUG)
        logging.info('Set the job ID to run {}'.format(setup_args['jobID']))
        train(**setup_args)
