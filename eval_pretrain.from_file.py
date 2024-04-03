"""Code to get embeddings from the checkpoint, using the provided files with labels as the guide
"""
import itertools
import json
import logging
import os
import uuid
from argparse import ArgumentParser
import torch.nn as nn

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

import model as model_module


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
    if 'output_folder' in kwargs:
        output_folder = kwargs['output_folder']
        _, cp_filename = os.path.split(cp_path)
    else:
        output_folder, cp_filename = os.path.split(cp_path)
    
    logging.basicConfig(
        filename=output_folder + '/eval_log.log', level=logging.DEBUG)
    logging.info("Output folder will be {0}".format(output_folder))

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

    logging.info("Running eval...")

    data_dir = './ds_train_hg38/pretrain_dataset'
    if 'data_dir' in kwargs:
        data_dir = kwargs['data_dir']
    seq_len_file = kwargs['seq_length_file']
    seq_len = kwargs['seq_length']

    if seq_len<seq_len_file:
        # we want a shorter sequence
        tail = (seq_len_file-seq_len)//2

    sampled_h5_file = os.path.join(data_dir, 'sc_sort.sampled_N=200k.seqlen={0}.h5'.format(seq_len_file))
    enh_h5_file = os.path.join(data_dir, 'F5.hg38.enhancers.seqlen={0}.h5'.format(seq_len_file))

    print(sampled_h5_file)
    sampled_predictions = []
    sampled_predictions_preproj = []
    with h5py.File(sampled_h5_file, 'r') as fh:
        seqs = fh['sequences']
        for s in seqs:
            s = torch.Tensor(s.reshape((1, s.shape[0], s.shape[1])))
            if seq_len<seq_len_file:
                s = s[:,tail:tail+seq_len,:]
            pred1, pred2 = model(s.float().permute(0, 2, 1).to(device))
            # specify what we want to save
            sampled_predictions.append(pred1.cpu().detach().numpy())
            sampled_predictions_preproj.append(pred2.cpu().detach().numpy())
    sampled_predictions = np.vstack(sampled_predictions)
    sampled_predictions_preproj = np.vstack(sampled_predictions_preproj)
    print(sampled_predictions.shape)
    np.save(os.path.join(output_folder, '{0}.sc_sort.sampled_N=200k.npy'.format(
        cp_filename)), sampled_predictions)
    np.save(os.path.join(output_folder, '{0}.preproj.sc_sort.sampled_N=200k.npy'.format(
        cp_filename)), sampled_predictions_preproj)
    del sampled_predictions
    del sampled_predictions_preproj

    print(enh_h5_file)
    enh_predictions = []
    enh_predictions_preproj = []
    with h5py.File(enh_h5_file, 'r') as fh:
        seqs = fh['sequences']
        for s in seqs:
            s = torch.Tensor(s.reshape((1, s.shape[0], s.shape[1])))
            if seq_len<seq_len_file:
                s = s[:,tail:tail+seq_len,:]
            pred1, pred2 = model(s.float().permute(0, 2, 1).to(device))
            enh_predictions.append(pred1.cpu().detach().numpy())
            enh_predictions_preproj.append(pred2.cpu().detach().numpy())
    enh_predictions = np.vstack(enh_predictions)
    enh_predictions_preproj = np.vstack(enh_predictions_preproj)
    print(enh_predictions.shape)
    np.save(os.path.join(output_folder, '{0}.F5.hg38.enhancers.npy'.format(
        cp_filename)), enh_predictions)
    np.save(os.path.join(output_folder, '{0}.preproj.F5.hg38.enhancers.npy'.format(
        cp_filename)), enh_predictions_preproj)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setup", help='A required .yaml file with the training setup')
    args = parser.parse_args()

    with open(args.setup) as f:
        setup_args = yaml.safe_load(f)

    eval(**setup_args)
