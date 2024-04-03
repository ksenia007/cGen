"""Code to get embeddings from the model for the sequences in our train file.
Used in linear probing - simple models on top of the embeddings downstream to compare performances.
Loops through all available checkpoints
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
    # print('checkpoint weights ', list(checkpoint_weights['state_dict'].keys()))
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
    Evaluate on the provided dataset
    """
    cp_path = kwargs['checkpoint_path']
    if 'output_path' in kwargs:
        output_folder = kwargs['output_path']
    else:
        output_folder = cp_path

    files = os.listdir(cp_path)
    files_filter = []
    for f in files:
        if ('.ckpt' not in f) or ('-v1' in f):
            print('Skipping', f)
            continue
    
        cp_filename = f
        print('Processing checkpoint:', cp_filename)
        checkpoint_path = cp_path+cp_filename
        print('checkpoint_path:', checkpoint_path)

        logging.basicConfig(
            filename=cp_path + '/eval_log.log', level=logging.DEBUG)
        logging.info("Output folder will be {0}".format(output_folder))

        epoch_value = cp_filename.split('-')[2]
        print('Epoch ', epoch_value)
    
        model_args = {
         'model_type': kwargs['model_type'],
         'model_name': kwargs['model_name'],
         'sequence_length': kwargs['seq_length'],
    }
        # allow used to specify shorter sequence
        if 'seq_length_use' in kwargs:
            model_args['sequence_length'] = kwargs['seq_length_use']
        else:
            model_args['sequence_length'] = kwargs['seq_length']

        if 'final_mlp_size' in kwargs:
            model_args['final_mlp_size'] = kwargs['final_mlp_size']
        
        model = model_module.ModelContrastive(**model_args)

        if torch.cuda.is_available():
            model = load_all_weights(model, checkpoint_path)
            #print(model)
            model.cuda()
        else:
            model = load_all_weights(model, checkpoint_path, cpu=True)
            #print(model)
            logging.info("Running on CPU")
        model.eval()

        print("Running eval...")

        files_list = ['train', 'valid', 'test']

        for f in files_list:

            data_dir = kwargs['data_dir'] #####
            data_file = kwargs['data_file']  #####
            data_file += f+'.h5'
            
            data_file_path = os.path.join(data_dir, data_file)
            print('using datafile',data_file_path)

            #targets_name = '_'.join(data_file.split('.')[:-2])
            targets_name = data_file.split('.')[-2]
            print('File targets', targets_name)
            
            sampled_predictions = []
            sampled_predictions_preproj = []

            seq_len_file = kwargs['seq_length_file']
            seq_len = kwargs['seq_length']

            if seq_len<seq_len_file:
                # we want a shorter sequence
                tail = (seq_len_file-seq_len)//2
            
            with h5py.File(data_file_path, 'r') as fh:
                # targets is what we predict
                seqs = fh['sequences']
                labels = np.array(fh['targets'])
                for s in seqs:
                    s = torch.Tensor(s.reshape((1, s.shape[0], s.shape[1])))
                    if seq_len<seq_len_file:
                        # we need to cut the sequence
                        #buffer_step = (kwargs['seq_length'] - kwargs['seq_length_use'])//2
                        #s = s[buffer_step:-buffer_step]
                        s = s[:,tail:tail+seq_len,:]
                    pred1, pred2 = model(s.float().permute(0, 2, 1).to(device))
                    # specify what we want to save
                    sampled_predictions.append(pred1.cpu().detach().numpy())
                    sampled_predictions_preproj.append(pred2.cpu().detach().numpy())
            
            sampled_predictions = np.vstack(sampled_predictions)
            sampled_predictions_preproj = np.vstack(sampled_predictions_preproj)
            print('sampled_predictions', sampled_predictions.shape)
            print('labels', labels.shape)
            np.save(os.path.join(output_folder, '{}_{}_encodings.npy'.format(
                targets_name, epoch_value)), sampled_predictions)
            np.save(os.path.join(output_folder, '{}_{}_encodings.preproj.npy'.format(
                targets_name, epoch_value)), sampled_predictions_preproj)
            np.save(os.path.join(output_folder, '{}_targets.npy'.format(
                targets_name)), labels)
            


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setup", help='A required .yaml file with the training setup')
    args = parser.parse_args()

    with open(args.setup) as f:
        setup_args = yaml.safe_load(f)

    eval(**setup_args)
