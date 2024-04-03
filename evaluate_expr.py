"""Code to evaluate the model.
"""

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
from dataloader import ExpressionData

from scipy import stats
from sklearn.metrics import r2_score


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_all_weights(model, checkpoint, cpu=False):
    """Function to load all the weights (strict load) for evaluation."""
    # load the weigths (with finetune layer)
    if cpu:
        checkpoint_weights = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint_weights = torch.load(checkpoint)
    print('checkpoint weights ', list(checkpoint_weights['state_dict'].keys()))
    for k in list(checkpoint_weights['state_dict'].keys()):
        if "finetune_layer" in k:
            new_key = k.replace("model.", "")
        else:
            new_key = k.replace(
                "model.backbone.backbone.", "backbone.backbone.")
            new_key = new_key.replace(
                "model.backbone.", "backbone.")
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
    if 'checkpointID' in setup_args:
        print('Checkpoint ID is specified')
        location_checkpoint = os.path.join(setup_args['checkpoint_loc'], setup_args['checkpointID'], 'checkpoints')
        files = os.listdir(location_checkpoint)
        files = [i for i in files if '.ckpt' in i]
        if len(files)==0:
            print('No checkpoints found', location_checkpoint)
            raise ValueError("No checkpoints found")
        if len(files)>1:
            found = False
            print('More than one checkpoint available, use CHECKPOINT to specify the correct one')
            print('Try to find -v1')
            for f in files:
                if 'v1' in f:
                    files = [f]
                    found = True
                    break
            if not found:
                print('v1 not found')
                print('Location checked: {}'.format(location_checkpoint))
                print('Files found', files)
                raise ValueError("Too many available checkpoints. See .out")
        
        cp_path = os.path.join(location_checkpoint, files[0])
        kwargs['checkpoint'] = cp_path
        print('*** Using checkpoint', cp_path)

    else:
        cp_path = kwargs['checkpoint']
    output_folder, cp_filename = os.path.split(cp_path)
    logging.basicConfig(
        filename=output_folder + '/eval_log.log', level=logging.DEBUG)
    logging.info("Output folder will be {0}".format(output_folder))

    if 'use_additional' in kwargs:
        use_additional = kwargs['use_additional']
    else:
        use_additional = None

    if 'additional_len' not in kwargs:
        kwargs['additional_len'] = 0
    if 'shift_noncoding' not in kwargs:
        kwargs['shift_noncoding'] = False
    if 'effective_length' not in kwargs:
        kwargs['effective_length'] = kwargs['seq_length']

    
    expr_data = ExpressionData('finetune',
                               seq_length=kwargs['seq_length'],
                               train_batch_size=kwargs['train_batch_size'],
                               eval_batch_size=kwargs['eval_batch_size'],
                               data_dir=kwargs['data_dir'], 
                               use_additional = use_additional,
                               h5_filepath = kwargs['h5_filepath'])
    expr_data.prepare_data()
    expr_data.setup()
    test_dataloader = expr_data.test_dataloader()
    logging.info("Prepared data!")

    if kwargs['task']=='TrainExpression':
        model = model_module.ModelExpression(model_type=kwargs["model_type"],
                                         model_name=kwargs['model_name'],
                                         sequence_length=kwargs['seq_length'],
                                         hidden_size=kwargs['hidden_size'],
                                         num_classes = kwargs['num_classes'], 
                                         checkpoint=kwargs['checkpoint'],
                         additional_len = int(kwargs['additional_len']),
                         shift_noncoding=kwargs['shift_noncoding'],
                         effective_length=kwargs['effective_length']
                                         )
    elif kwargs['task']=='TrainBinaryExpression':
        model = model_module.ModelBinaryExpression(model_type=kwargs["model_type"],
                                         model_name=kwargs['model_name'],
                                         sequence_length=kwargs['seq_length'],
                                         hidden_size=kwargs['hidden_size'],
                                         num_classes = kwargs['num_classes'])
    model = load_all_weights(model, kwargs['checkpoint'])
    print(model)
    logging.info('First finetune weights:', model.finetune_layer[0].weight[0][0:3])

    if torch.cuda.is_available():
        model.cuda()
    else:
        logging.info("Running on CPU")
    model.eval()

    logging.info("Running eval...")
    labels = []
    predictions = []

    if use_additional:
        print('Use additional inputs')
        for seq, label, additional in test_dataloader:
            pred = model(seq.float().permute(0, 2, 1).to(device), additional.to(device))
            # label = torch.log10(torch.add(label, 0.1))
            labels.append(label.cpu().detach().numpy())
            predictions.append(pred.cpu().detach().numpy())
    else:
        for seq, label in test_dataloader:
            pred = model(seq.float().permute(0, 2, 1).to(device))
            # label = torch.log10(torch.add(label, 0.1))
            labels.append(label.cpu().detach().numpy())
            predictions.append(pred.cpu().detach().numpy())
    labels = np.vstack(labels)
    predictions = np.vstack(predictions)
    np.save(os.path.join(output_folder, 'testset_predictions.seqlen={0}.npy'.format(
                kwargs['seq_length'])),
            predictions)
    np.save(os.path.join(output_folder, 'testset_labels.seqlen={0}.npy'.format(
                kwargs['seq_length'])),
            labels)
    print('**********')
    print('SPEARMAN', stats.spearmanr(labels.flatten(), predictions.flatten()))
    print('R2: ', r2_score(labels.flatten(), predictions.flatten()))
    print('**********')
    print(predictions.flatten())
    print('**********')
    print(labels.flatten())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setup", help='A required .yaml file with the training setup')
    args = parser.parse_args()

    with open(args.setup) as f:
        setup_args = yaml.safe_load(f)

    eval(**setup_args)
