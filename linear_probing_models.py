import os
import sys
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression



import gc

loc = str(sys.argv[1])


def get_avail_epochs(loc):
    """Get list of available epochs"""
    files = os.listdir(loc)
    epochs = []
    for f in files:
        if 'epoch' and 'encodings' in f:
            epochs.append(f.split('_')[1].split('=')[-1])
    epochs = list(set(epochs))
    return epochs

def get_epoch_file(epoch_n, loc, use_preproj = True):
    """GIven epoch number (name, could be '00') get files with embeddings"""
    
    if use_preproj:
        file_valid = 'valid_epoch={}_encodings.preproj.npy'.format(epoch_n)
        file_train = 'train_epoch={}_encodings.preproj.npy'.format(epoch_n)
    else:
        file_valid = 'valid_epoch={}_encodings.npy'.format(epoch_n)
        file_train = 'train_epoch={}_encodings.npy'.format(epoch_n)
    
    targets_train = 'train_targets.npy'
    targets_valid = 'valid_targets.npy'
    
    x = np.load(loc+file_train)
    y = np.load(loc+targets_train)

    x_val = np.load(loc+file_valid)
    y_val = np.load(loc+targets_valid)

    
    if use_preproj:  
        x = x.reshape(x.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
    
    return x, y, x_val, y_val

def fit_model(x, y_feat, x_val, y_val_feat):
    """Fit the model to one feature at a time"""
    
    scores = {}
    reg = LinearRegression().fit(x, y_feat)
    preds = reg.predict(x_val)
    
    s, p = spearmanr(preds, y_val_feat)
    scores['spearman'] = s
    
    return s, preds #scores

def predict_all_features(x, y, x_val, y_val):
    """For a given epoch, get predictions for all columns in y (record idx)"""
    feature_results = {}
    preds_array = {}
    for i in range(y.shape[1]):
        print(i)
        s, preds = fit_model(x, y[:,i], x_val, y_val[:,i])
        feature_results[i] = s
        preds_array[i] = preds
        gc.collect()
    return feature_results, preds_array

def predict_many_epochs_one_feat(epoch_list, loc, feat_idx, use_preproj = True):
    """For a given feature idx (column) get predictions for given epochs"""
    epoch_results = {}
    for e in epoch_list:
        x, y, x_val, y_val = get_epoch_file(e, loc, use_preproj)
        s, _ = fit_model(x, y[:,feat_idx], x_val, y_val[:,feat_idx])
        epoch_results[int(e)] = s
        del x, y, x_val, y_val
        gc.collect()
    return epoch_results

def predict_many_epochs_all_feat(epoch_list,  loc, use_preproj = True):
    """For a given job ID & epoch list get predictions for all features"""
    all_results = {}
    preds_all = {}
    for e in epoch_list:
        print(e)
        x, y, x_val, y_val = get_epoch_file(e, loc, use_preproj)
        spearmans, preds = predict_all_features(x, y, x_val, y_val)
        all_results[int(e)] = spearmans
        preds_all[int(e)] = preds
        del x, y, x_val, y_val
        gc.collect()
    return all_results, preds_all


have_epochs = get_avail_epochs(loc)
print(have_epochs)
have_epochs = [int(i) for i in have_epochs]
have_epochs.sort()
have_epochs = have_epochs[:60]
have_epochs = have_epochs[::4]
have_epochs = [str(i) if len(str(i))>1 else '0'+str(i) for i in have_epochs]
print('RUN FOR:', have_epochs)

many_epochs_job, preds_all = predict_many_epochs_all_feat(have_epochs, loc, use_preproj = False)
many_epochs_job = pd.DataFrame(many_epochs_job)
many_epochs_job.to_csv(loc+'every3_epochs_spearman.postprojection.csv')
preds_all = pd.DataFrame(preds_all)
preds_all.to_csv(loc+'every3_epochs_PREDS.postprojection.csv')