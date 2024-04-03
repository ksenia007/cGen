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
import model as model_module
import dataloader as dl_module
from simCGen import simCGen


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def process_transforms(**kwargs):
    """Create a list of transforms"""
    trans = []
    for i, v in kwargs['augments'].items():
        trans_function = getattr(aug, i)

        if not v:
            v = {}  # possible that there are no params if we only need seq_length and buffer

        if 'seq_length_use' in kwargs:
            print('replacing max sequence {} length with the requested {}'.format(kwargs['seq_length'], kwargs['seq_length_use']))
            v['seq_length'] = kwargs['seq_length_use']
        else:
            v['seq_length'] = kwargs['seq_length']
        if 'max_buffer_use' in kwargs:
            v['buffer'] = kwargs['max_buffer_use']
            print('Max buffer provided, use buffer', v['buffer'])
        else:
            v['buffer'] = kwargs['buffer']

        trans.append(trans_function(**v))

    trans.append(transforms.ToTensor())

    return aug.ContrastiveTransformations(transforms.Compose(trans))


def train(**kwargs):
    # define dataset and augmentations
    contrast_transforms = process_transforms(**kwargs)

    if 'dataloader' in kwargs:
        dl_class = getattr(dl_module, kwargs['dataloader'])
    else:
        dl_class = getattr(dl_module, 'GenomicData')
    print(dl_class)

    dl_args = {
        'model_name_or_path': '',
        'seq_length': kwargs['seq_length'] + 2*kwargs['buffer'],
        'transform': contrast_transforms,
        'train_batch_size': kwargs['train_batch_size'],
        'eval_batch_size': kwargs['eval_batch_size'],
        'data_dir': kwargs['data_dir'],
        'n_cpus': kwargs['n_cpus'],
    }

    if 'use_additional' in kwargs:
        dl_args['use_additional'] = kwargs['use_additional']

    data = dl_class(**dl_args)
    data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    logging.info("Prepared data!")

    # checkpoint
    checkpoint_last = ModelCheckpoint(
        save_top_k = -1,
        every_n_epochs = 1,
        #mode='max', monitor='epoch',
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
    lr_monitor = LearningRateMonitor(
        logging_interval='step')

    # define train
    # ModelContrastive or ModelFinetune
    backbone = getattr(model_module, kwargs['backbone'])
    if ('seq_length_use' in kwargs) and (kwargs['seq_length_use']!=kwargs['seq_length']):
        backbone = backbone(
            sequence_length=kwargs['seq_length_use'],
            model_type=kwargs["model_type"],  # Albert, Nystromformer
            model_name=kwargs['model_name'],
            final_mlp_size=kwargs['final_mlp_size'],
        )
        # print('replaxing max sequence length with the actual')
        kwargs['seq_length'] = kwargs['seq_length_use']
    else: 
        backbone = backbone(
            sequence_length=kwargs['seq_length'],
            model_type=kwargs["model_type"],  # Albert, Nystromformer
            model_name=kwargs['model_name'],
            final_mlp_size=kwargs['final_mlp_size'],
        )
    model = simCGen(lr=kwargs['lr'], temperature=kwargs['temperature'],
                   weight_decay=kwargs['weight_decay'], backbone=backbone,
                   limit_train_batches=kwargs['trainer']['limit_train_batches'],
                   max_epochs=kwargs['trainer']['max_epochs'])
    
    logger = TensorBoardLogger(kwargs['tensorboard_folder'], name="")

    trainer_args = kwargs['trainer']
    if 'accelerator' not in trainer_args:
        trainer_args['accelerator'] = 'auto'
    trainer = pl.Trainer(logger=logger,
                         callbacks=[checkpoint_last,
                                    checkpoint_lowest,
                                    lr_monitor, ResetSampler()],
                         **trainer_args)
    logging.info("Set up training, starting to train")

    # train
    if 'checkpoint' in kwargs:
        trainer.fit(model=model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=kwargs['checkpoint'])
    else:
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

    # create a nested folder structure
    if 'output_folder' not in setup_args:
        setup_args['output_folder'] = 'outputs/'+id_job
    else:
        setup_args['output_folder'] = os.path.join(
            setup_args['output_folder'], id_job)

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

    # start logging session
    logging.basicConfig(
        filename=setup_args['output_folder']+'/log.log', level=logging.DEBUG)
    logging.info('Set the job ID to run {}'.format(setup_args['jobID']))

    train(**setup_args)
