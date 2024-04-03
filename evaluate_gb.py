"""Code to evaluate the model.
"""

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
from torchmetrics import AUROC
from torchmetrics import AveragePrecision
import yaml

import model as model_module
from dataloader import GenomicData

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from selene_sdk.utils.performance_metrics import compute_score

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _calculate_auroc(preds, labels):
    """Not used currently, remove when finalized"""
    labels = labels.int()
    auroc = AUROC(num_classes=labels.size()[1])
    auc = auroc(preds, labels)
    return auc


def _calculate_avg_precision(preds, labels):
    """Not used currently, remove when finalized"""
    labels = labels.int()
    avg_precision = AveragePrecision(num_classes=labels.size()[1])
    ap = avg_precision(preds, labels)
    return ap

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
            #new_key = new_key.replace('3', '2')
            new_key = new_key.replace('2', '3')
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
    cp_path = kwargs['checkpoint']
    output_folder, cp_filename = os.path.split(cp_path)
    print(output_folder, cp_filename)
    logging.basicConfig(
        filename=output_folder + '/eval_log.log', level=logging.DEBUG)
    logging.info("Output folder will be {0}".format(output_folder))

    dl_args = {
        'seq_length': kwargs['seq_length'],
        'train_batch_size': kwargs['train_batch_size'],
        'eval_batch_size': kwargs['eval_batch_size'],
    }
    if 'n_cpus' in kwargs:
        dl_args['n_cpus'] = kwargs['n_cpus']

    logging.info("Prepared data!")

    model_args = {
        'model_type': kwargs['model_type'],
        'model_name': kwargs['model_name'],
        'sequence_length': kwargs['seq_length'],
        'num_classes': kwargs['num_classes']
    }
    if 'hidden_size' in kwargs:
        model_args['hidden_size'] = kwargs['hidden_size']

    model = model_module.ModelChromProfiling(**model_args)
    model = load_all_weights(model, kwargs['checkpoint'])
    print(model)
    logging.info('First finetune weights:', model.finetune_layer[0].weight[0][0:3])

    if torch.cuda.is_available():
        model.cuda()
    else:
        logging.info("Running on CPU")
    model.eval()

    print(kwargs['checkpoint'])
    logging.info("Running eval...")

    testfile = os.path.join('./genomic_benchmarks', kwargs['dataset'], 'test_dataset.h5')
    print(testfile)
    sequences = h5py.File(testfile, 'r')['sequences'][()]
    labels = h5py.File(testfile, 'r')['targets'][()]
    print(sequences.shape, labels.shape)

    predictions = []
    loss_fn = nn.BCELoss()
    for ix in range(0, len(sequences), kwargs['eval_batch_size']):
        s = ix
        e = min(ix+kwargs['eval_batch_size'], len(sequences))
        seq = sequences[s:e]
        seq = torch.Tensor(seq).float()
        pred = model(seq.permute(0, 2, 1).to(device))
        if len(labels)<2:
            print('preds1', pred)
        predictions.append(pred.cpu().detach().numpy())
        #if len(labels) * label.size()[0] >= N_samples:
        #    break


    predictions = np.vstack(predictions)
    print(predictions.shape, labels.shape)
    np.save(os.path.join(output_folder, '{0}.predictions.npy'.format(cp_filename)), predictions)

    avg_auc, aucs = compute_score(predictions, labels, roc_auc_score,
                                  report_gt_feature_n_positives=1)
    avg_precision, avg_precs = compute_score(predictions, labels, average_precision_score,
                                             report_gt_feature_n_positives=1)
    logging.info("AUC: {0}, Avg Precision: {1}".format(avg_auc, avg_precision))
    logging.info("Outputting AUC and Avg Precision per-chromatin-profile to {0}".format(
        output_folder))
    np.save(os.path.join(output_folder, '{0}.aucs.N={1}.test_fromsave.npy'.format(
        cp_filename, len(labels))), aucs)
    np.save(os.path.join(output_folder, '{0}.avg_precs.N={1}.test_fromsave.npy'.format(
        cp_filename, len(labels))), avg_precs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--setup", help='A required .yaml file with the training setup')
    args = parser.parse_args()

    with open(args.setup) as f:
        setup_args = yaml.safe_load(f)

    eval(**setup_args)
