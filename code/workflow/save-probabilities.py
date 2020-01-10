# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Write summary pf evaluation results to files in the output directory."""

import json
import logging
import numpy as np
import os
import pickle
import scipy as sp
import statistics
import sys

import files as fn
import recnn.utils as utils


# -- Helper Functions ----------------------------------------------------------

def load_results(data_dir, result_dir, verbose=False):
    """Load all the info for each run in a dictionary (hyperparameteres, auc,
    fpr, tpr, output prob, etc).
    """
    # For each folder in the result directory that starts with the prefix 'run_'
    # and contains a metrics resutl file we read the results as well as the
    # parameters from the respective folder in the data directory.
    results = list()
    for subdir in os.listdir(result_dir):
        if subdir.startswith(fn.RUN_DIR_PREFIX):
            run_dir = os.path.join(result_dir, subdir)
            result_file = os.path.join(run_dir, fn.METRICS_FILE)
            if os.path.isfile(result_file):
                # If the result file exists we assume that the probabilities
                # file also exists as well as the run parameters file
                with open(result_file, 'r') as f:
                    data = json.load(f)
                # Read output probabilities and true values
                y_prob_file  = os.path.join(run_dir, fn.Y_PROB_TRUE_FILE)
                with open(y_prob_file, 'rb') as f:
                    y_prob_true = list(pickle.load(f))
                # Read run parameters
                params_file = os.path.join(data_dir, subdir, fn.PARAMS_FILE)
                params = utils.Params(params_file)
                dictionary = {
                    'runName': subdir,
                    'lr': params.learning_rate,
                    'decay': params.decay,
                    'batch': params.batch_size,
                    'hidden': params.hidden,
                    'Njets': params.myN_jets,
                    'Nfeatures': params.features,
                    'accuracy': data['accuracy'],
                    'loss': data['loss'],
                    'auc': data['auc'],
                    'yProbTrue': np.asarray(y_prob_true)
                }
                results.append(dictionary)
                if verbose:
                    logging.info(dictionary)
    return results


# -- Main Function -------------------------------------------------------------

if __name__ == '__main__':
    # Get command line arguments. Expects a reference to the data base directory
    # that contains the run parameters as well as the output directory that
    # contains the run results.
    args = sys.argv[1:]
    if len(args) != 2:
        print('Usage: {} <data-dir> <result_dir>'.format(sys.argv[0]))
        sys.exit(-1)
    data_dir = args[0]
    results_dir = args[1]
    # Initialize the logger
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)
    # Parse result files and generate summary of outputs
    results = load_results(data_dir, results_dir, verbose=True)
    y_prob_best = np.asarray([element['yProbTrue'] for element in results])
    # Saving output probabilities and true values
    output_file = os.path.join(results_dir, fn.Y_PROB_BEST_FILE)
    logging.info('Saving output probabilities and true values to {}'.format(output_file))
    with open(output_file, 'wb') as f:
        pickle.dump(y_prob_best[:,:,0], f)
    # Create Json file containing an array of accuracy, loss, and auc values
    # for each run
    accuracy = list()
    loss = list()
    auc = list()
    values = list()
    for r in results:
        values.append({
            'accuracy': r['accuracy'],
            'loss': r['loss'],
            'auc': r['auc']
        })
        accuracy.append(r['accuracy'])
        loss.append(r['loss'])
        auc.append(r['auc'])
    doc = {
        'accuracy': {
            'min': min(accuracy),
            'max': max(accuracy),
            'mean': statistics.mean(accuracy),
            'stdev': statistics.stdev(accuracy)
        },
        'loss': {
            'min': min(loss),
            'max': max(loss),
            'mean': statistics.mean(loss),
            'stdev': statistics.stdev(loss)
        },
        'auc': {
            'min': min(auc),
            'max': max(auc),
            'mean': statistics.mean(auc),
            'stdev': statistics.stdev(auc)
        },
        'results': values
    }
    output_file = os.path.join(results_dir, fn.RESULT_FILE)
    logging.info('Saving result summary to {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(doc, f, indent=4)
