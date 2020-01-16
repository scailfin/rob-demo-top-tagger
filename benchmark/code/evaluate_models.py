# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Default implementation for the model evaluation step in the Top Tagger
workflow.

usage: evaluate-models.py [-h] [-a ARCHITECTURE] [-r RESTORE] [-s N_START]
                          [-f N_FINISH]
                          tree_file data_dir output_dir
"""

import argparse
import json
import logging
import numpy as np
import os
import pickle
import sys
import time

import files as fn
import recnn.evaluate as eval
import recnn.utils as utils


# -- Helper Functions ---------------------------------------------------------

def load_results(data_dir, result_dir, n_start, n_finish, verbose=False):
    """Load all the info for each run in a dictionary (hyperparameteres, auc,
    fpr, tpr, output prob, etc).
    """
    # For each folder in the result directory that starts with the prefix
    # 'run_' and contains a metrics resutl file we read the results as well as
    # the parameters from the respective folder in the data directory.
    results = list()
    for subdir in os.listdir(result_dir):
        if subdir.startswith(fn.RUN_DIR_PREFIX):
            run_id = int(subdir[subdir.rfind('_') + 1])
            if n_start <= run_id < n_finish:
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
                        'yProbTrue': list(y_prob_true)
                    }
                    results.append(dictionary)
                    if verbose:
                        logging.info(dictionary)
    return results


# -- Main Function ------------------------------------------------------------

def main(args):
    # Ensure that the output directory exists
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # Initialize the logger
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(args.output_dir, fn.EVAL_LOG_FILE)
    log.addHandler(logging.FileHandler(log_file))
    logging.getLogger().setLevel(logging.DEBUG)
    # Call evaluation function for each model
    start_time = time.time()
    for n_run in np.arange(args.n_start, args.n_finish):
        run_id = '{}{}'.format(fn.RUN_DIR_PREFIX, n_run)
        run_dir = os.path.join(args.data_dir, run_id)
        eval.run(
            tree_file=args.tree_file,
            params=utils.Params(os.path.join(run_dir, fn.PARAMS_FILE)),
            architecture=args.algorithm,
            restore_file=os.path.join(run_dir, '{}.pth.tar'.format(args.restore)),
            output_dir=os.path.join(args.output_dir, run_id)
        )
    # Combine result files to generate output file
    logging.info('Combine runs {}-{}'.format(args.n_start, args.n_finish))
    results = load_results(
        data_dir=args.data_dir,
        result_dir=args.output_dir,
        n_start=args.n_start,
        n_finish=args.n_finish,
        verbose=False
    )
    y_prob_best = np.asarray([element['yProbTrue'] for element in results])
    # Saving output probabilities and true values
    output_file = os.path.join(args.output_dir, fn.Y_PROB_BEST_FILE)
    msg = 'Saving output probabilities and true values to {}'
    logging.info(msg.format(output_file))
    with open(output_file, 'wb') as f:
        #pickle.dump(y_prob_best[:, :, 0], f)
        pickle.dump(results, f)
    # Log runtime information
    exec_time = time.time()-start_time
    logging.info('Preprocessing time (minutes) = {}'.format(exec_time/60))


if __name__ == '__main__':
    # Get command line parameters. The model evaluator expects a reference to
    # the tree dataset that was generated by the preprocessing step, the data
    # directory that contains the evaluation run parameter files, and the
    # outout base directory for run.
    # There are four optional command line arguments:
    # - algorithm (default: )
    # - restore (default: best)
    # - n_start (default: 0)
    # - n_finish (default: 9)
    parser = argparse.ArgumentParser(
        description='Evaluate models for trained weights.'
    )
    parser.add_argument(
        '-a', '--algorithm',
        default='NiNRecNNReLU',
        help='Model architecture identifier'
    )
    parser.add_argument(
        '-r', '--restore',
        default='best',
        help='Restore strategy name'
    )
    parser.add_argument(
        '-s', '--n_start',
        type=int,
        default=0,
        help='Start model number.'
    )
    parser.add_argument(
        '-f', '--n_finish',
        type=int,
        default=9,
        help='End model number.'
    )
    parser.add_argument('tree_file', help='Preprocessed tree dataset.')
    parser.add_argument('data_dir', help='Base directory for run parameters.')
    parser.add_argument('output_dir', help='Base directory for run results.')
    main(args=parser.parse_args())