# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Evaluates the model"""

import argparse
import logging
import os
import numpy as np
import torch
import sys
import time
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn.utils import check_random_state

import model.data_loader as dl
from model import recNet as net
import utils
from model import preprocess
import model.dataset as dataset
from scipy import interp


#-- Evaluation Functions -------------------------------------------------------

def generate_results(y_test, y_score, batch_size, output_file, weights=None):
    """Make ROC with area under the curve plot."""
    logging.info('length y_test={}'.format(len(y_test)))
    logging.info('Lenght y_score={}'.format(len(y_score)))
    if weights is not None:
        logging.info('Sample weights length={}'.format(len(weights)))
        # We include the weights until the last full batch (the remaining ones
        # are not enough to make a full batch)
        last_weight = len(weights) - len(weights) % batch_size
        weights = weights[0:last_weight]
        logging.info('New sample weights length={}'.format(len(weights)))
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_score,
        pos_label=1,
        drop_intermediate=False
    )
    logging.info('Length y_score {}'.format(len(y_score)))
    logging.info('Length y_test {}'.format(len(y_test)))
    logging.info('Thresholds[0:6] = \n {}'.format(thresholds[:6]))
    logging.info('Thresholds lenght = \n{}'.format(len(thresholds)))
    logging.info('fpr lenght{}'.format(len(fpr)))
    logging.info('tpr lenght{}'.format(len(tpr)))
    logging.info('Sample weights length={}'.format(len(weights)))
    logging.info('Sample weights[0:4]={}'.format(weights[0:4]))
    # Save fpr, tpr to output file
    with open(output_file, 'wb') as f:
        pickle.dump(zip(fpr,tpr), f)
    roc_auc = roc_auc_score(y_test, y_score)
    logging.info('roc_auc={}'.format(roc_auc))
    return roc_auc


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, output_file, sample_weights=None):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network superclass
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to evaluation mode
    model.eval()
    # summary for current eval loop
    summ = []
    out_prob=[]
    labels=np.array([])
    # compute metrics over the dataset
    data_iterator_iter = iter(data_iterator)
    for _ in range(num_steps):
        # fetch the next evaluation batch
        levels, children, n_inners, contents, n_level, labels_batch=next(data_iterator_iter)
        # shift tensors to GPU if available
        if params.cuda:
            levels = levels.cuda()
            children=children.cuda()
            n_inners=n_inners.cuda()
            contents=contents.cuda()
            n_level= n_level.cuda()
            labels_batch =labels_batch.cuda()
        # convert them to Variables to record operations in the computational graph
        levels=torch.autograd.Variable(levels)
        children=torch.autograd.Variable(children)
        n_inners=torch.autograd.Variable(n_inners)
        contents = torch.autograd.Variable(contents)
        n_level=torch.autograd.Variable(n_level)
        labels_batch = torch.autograd.Variable(labels_batch)
        # Feedforward pass through the NN
        output_batch = model(params, levels, children, n_inners, contents, n_level)
        # compute model output
        labels_batch = labels_batch.float() #Uncomment if using torch.nn.BCELoss() loss function
        output_batch=output_batch.view((params.batch_size))
        loss = loss_fn(output_batch, labels_batch)
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        labels=np.concatenate((labels,labels_batch))
        out_prob=np.concatenate((out_prob,output_batch))
        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
    # Get the bg rejection at 30% tag eff: 0.05 + 125*(1 - 0.05)/476=0.3).
    # That's why we pick 125
    fpr_log, tpr_log, thresholds_log = roc_curve(
        labels,
        out_prob,
        pos_label=1,
        drop_intermediate=False
    )
    base_tpr = np.linspace(0.05, 1, 476)
    inv_fpr = interp(base_tpr, tpr_log, 1. / fpr_log)[125]
    logging.info('Total Labels={}'.format(labels[0:10]))
    logging.info('Out prob={}'.format(out_prob[0:10]))
    logging.info('------------'*10)
    logging.info('len labels after ={}'.format(len(labels)))
    logging.info('len out_prob after{}'.format(len(out_prob)))
    # Get fpr, tpr, ROC curve and AUC
    roc_auc = generate_results(
        y_test=labels,
        y_score=out_prob,
        batch_size=params.batch_size,
        output_file=output_file,
        weights=sample_weights
    )
    # Save output prob and true values
    with open(os.path.join(output_dir, 'yProbTrue.pkl', 'wb') as f:
        pickle.dump(zip(out_prob, labels), f)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    metrics_mean['auc']=roc_auc
    metrics_mean['test_bg_reject']=inv_fpr
    return metrics_mean


# -- Main Function -------------------------------------------------------------

def run(input_file, params, architecture, restore_file, output_dir):
    # Start pre-processing job. Main code block with the methods to load the
    # raw data, create and preprocess the trees.
    logging.info('Preprocessing jet trees ...')
    start_time = time.time()


    cmd_eval = "CUDA_VISIBLE_DEVICES={gpu} {python} evaluate.py --model_dir={model_dir} --data_dir={data_dir} --sample_name={sample_name} --jet_algorithm={algo} --architecture={architecture} --restore_file={restore_file}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=eval_data_dir,sample_name=sample_name, algo=algo, architecture=architecture, restore_file=restore_file)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    # Set the random seed for reproducible experiments
    if params.cuda:
        torch.cuda.seed()
    # Main class with the methods to load the raw data and create the batches
    data_loader=dl.DataLoader
    ## Load batches of test data
    logging.info('Loading the dataset {}'.format(input_file))
    with open(input_file, 'rb') as f:
        test_data = pickle.load(f)
    # Architecture. Define the model and optimizer
    if architecture=='simpleRecNN':
        model = net.PredictFromParticleEmbedding(
            params,
            make_embedding=net.GRNNTransformSimple
        )
    elif architecture=='gatedRecNN':
        model = net.PredictFromParticleEmbeddingGated(
            params,
            make_embedding=net.GRNNTransformGated
        )
    elif architecture=='leaves_inner_RecNN':
        model = net.PredictFromParticleEmbeddingLeaves(
            params,
            make_embedding=net.GRNNTransformLeaves
        )
    elif architecture=='NiNRecNN':
        model = net.PredictFromParticleEmbeddingNiN(
            params,
            make_embedding=net.GRNNTransformSimpleNiN
        )
    elif architecture=='NiNRecNN2L3W':
        model = net.PredictFromParticleEmbeddingNiN2L3W(
            params,
            make_embedding=net.GRNNTransformSimpleNiN2L3W
        )
    elif architecture=='NiNgatedRecNN':
        model = net.PredictFromParticleEmbeddingGatedNiN(
            params,
            make_embedding=net.GRNNTransformGatedNiN
        )
    elif architecture=='NiNRecNNReLU':
        model = net.PredictFromParticleEmbeddingNiNReLU(
            params,
            make_embedding=net.GRNNTransformSimpleNiNReLU
        )
    else:
        raise ValueError('unknown architecture {}'.format(architecture))
    if params.cuda:
        model = model.cuda()
    # Loss function
    loss_fn = torch.nn.BCELoss()
    metrics = net.metrics
    #
    # EVALUATE
    #
    logging.info("Starting evaluation")
    # Reload weights from the saved file
    utils.load_checkpoint(restore_file, model)
    test_data = list(test_data)
    num_steps_test = len(test_data)//params.batch_size
    logging.info('num_steps_test={}'.format(num_steps_test))
    # We get an integer number of batches
    test_x = np.asarray([x for (x,y) in test_data][0:num_steps_test*params.batch_size])
    test_y = np.asarray([y for (x,y) in test_data][0:num_steps_test*params.batch_size])
    # Create tain and val datasets. Customized dataset class dataset.TreeDataset
    # that will create the batches by calling data_loader.batch_nyu_pad.
    test_data = dataset.TreeDataset(
        data=test_x,
        labels=test_y,
        transform=data_loader.batch_nyu_pad,
        batch_size=params.batch_size,
        features=params.features,
        shuffle=False
    )
    # Create the dataloader for the train and val sets (default Pytorch dataloader). Paralelize the batch generation with num_workers. BATCH SIZE SHOULD ALWAYS BE = 1 (batches are only loaded here as a single element, and they are created with dataset.TreeDataset).
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.customized_collate
    )
    # Evaluate the model
    test_metrics = evaluate(
        model=model,
        loss_fn=loss_fn,
        data_iterator=test_loader,
        metrics=metrics,
        params=params,
        output_file=output_file,
        num_steps=num_steps_test
    )
    utils.save_dict_to_json(test_metrics, output_file)

    # Log runtime information
    exec_time = time.time()-start_time
    logging.info('Preprocessing time (minutes) = {}'.format(exec_time/60))
