"""Code to evaluate the model.
"""
import itertools
import json
import logging
import os
import uuid
from argparse import ArgumentParser
import torch.nn as nn

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

import model as model_module
from dataloader import GenomicData

from selene_sdk.samplers import RandomPositionsSampler


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_all_weights(model, checkpoint, cpu=False):
    """Function to load all the weights (strict load) for evaluation."""
    if cpu:
        checkpoint_weights = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint_weights = torch.load(checkpoint)
    print('checkpoint weights ', list(checkpoint_weights['state_dict'].keys()))
    for k in list(checkpoint_weights['state_dict'].keys()):
        new_key = k.replace(
            "backbone.backbone.", "backbone.")
        new_key = new_key.replace(
            "backbone.contr_layer.", "contr_layer.")
        new_key = new_key.replace(
                    "backbone.backbone.backbone.", "backbone.")
        checkpoint_weights['state_dict'][new_key] = checkpoint_weights['state_dict'].pop(
            k)
    model.load_state_dict(checkpoint_weights['state_dict'], strict=True)
    return model


def eval(**kwargs):
    """
    Evaluate on randomly sampled validation set for now.
    """
    cp_path = kwargs['checkpoint']
    output_folder, cp_filename = os.path.split(cp_path)
    logging.basicConfig(
        filename=output_folder + '/eval_log.log', level=logging.DEBUG)
    logging.info("Output folder will be {0}".format(output_folder))

    dl_args = {
         'seq_length': kwargs['seq_length'],
         'train_batch_size': kwargs['train_batch_size'],
         'eval_batch_size': kwargs['eval_batch_size'],
    }
    if 'data_dir' in kwargs:
     dl_args['data_dir'] = kwargs['data_dir']
    if 'targets_file' in kwargs:
     dl_args['targets_file'] = kwargs['targets_file']
    if 'txt_file' in kwargs:
     dl_args['txt_file'] = kwargs['txt_file']
    if 'n_cpus' in kwargs:
     dl_args['n_cpus'] = kwargs['n_cpus']
    dl_args['return_coords'] = True

    genomic_data = GenomicData('pretrain', **dl_args)
    genomic_data.prepare_data()
    genomic_data.setup()
    train_dataloader = genomic_data.train_dataloader()
    logging.info("Prepared data!")

    model_args = {
         'model_type': kwargs['model_type'],
         'model_name': kwargs['model_name'],
         'sequence_length': kwargs['seq_length'],
    }
    if 'final_mlp_size' in kwargs:
        model_args['final_mlp_size'] = kwargs['final_mlp_size']
    model = model_module.ModelContrastive(**model_args)

    if torch.cuda.is_available():
        model = load_all_weights(model, kwargs['checkpoint'])
        print(model)
        model.cuda()
    else:
        model = load_all_weights(model, kwargs['checkpoint'], cpu=True)
        print(model)
        logging.info("Running on CPU")
    model.eval()
    N_samples = kwargs['n_samples']

    logging.info("Running eval...")
    coordset = set()
    repeats = 0

    coords = []
    predictions = []
    loss_fn = nn.BCELoss()
    for seq, label, coord in train_dataloader:
        pred = model(seq.float().permute(0, 2, 1).to(device))
        for c in coord[0]:
            if c not in coordset:
                coords.append(c)
                coordset.add(c)
            else:
                repeats += 1
        predictions.append(pred.cpu().detach().numpy())
        if len(coords) >= N_samples:
            break
    print(coords[:5])
    print(len(coords))
    print(repeats)
    predictions = np.vstack(predictions)

    np.save(os.path.join(output_folder, '{0}.N={1}.predictions.npy'.format(
        cp_filename, len(coords))), predictions)
    np.save(os.path.join(output_folder, '{0}.N={1}.coordinates.npy'.format(
        cp_filename, len(coords))), coords)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setup", help='A required .yaml file with the training setup')
    args = parser.parse_args()

    with open(args.setup) as f:
        setup_args = yaml.safe_load(f)

    eval(**setup_args)
