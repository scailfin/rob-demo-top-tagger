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
import sys

import files as fn
import recnn.analyze as analyze


# -- Main Function ------------------------------------------------------------

if __name__ == '__main__':
    # Get command line arguments. Expects a reference to the data base
    # directory that contains the run parameters as well as the output
    # directory that contains the run results.
    args = sys.argv[1:]
    if len(args) != 2:
        print('Usage: {} <data-dir> <result_dir>'.format(sys.argv[0]))
        sys.exit(-1)
    data_dir = args[0]
    results_dir = args[1]
    # Initialize the logger
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(results_dir, fn.ANALYZE_LOG_FILE)
    log.addHandler(logging.FileHandler(log_file))
    log.setLevel(logging.DEBUG)
    # Reading output probabilities and true values
    result_file = os.path.join(results_dir, fn.Y_PROB_BEST_FILE)
    msg = 'Reading output probabilities from {}'
    logging.info(msg.format(result_file))
    with open(result_file, 'rb') as f:
        y_prob_best = pickle.load(f)
    # Load groundtruth labels. Keep only the labels that come after the start
    # index
    with open(os.path.join(data_dir, 'labels.pkl'), 'rb') as f:
        true_labels = np.asarray(pickle.load(f), dtype=int)
    true_labels = true_labels[400000:]
    (
        bg_reject_outliers,
        bg_reject_std_outliers,
        aucs_outliers,
        auc_std_outliers
    ) = analyze.get_median_bg_reject(y_prob_best, true_labels)
    doc = {
        'bg_reject_outliers': bg_reject_outliers[0],
        'bg_reject_std_outliers': bg_reject_std_outliers[0],
        'aucs_outliers': aucs_outliers[0],
        'auc_std_outliers': auc_std_outliers[0]
    }
    output_file = os.path.join(results_dir, fn.RESULT_FILE)
    logging.info(json.dumps(doc, indent=4))
    logging.info('Saving result summary to {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(doc, f, indent=4)
