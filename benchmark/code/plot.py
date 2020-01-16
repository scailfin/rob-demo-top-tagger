# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def get_median_model(true_labels, tagger_outputs):
    """Get median model sorted by auc."""
    taggers_median = dict()
    for t_name in tagger_outputs:
        out_probs = tagger_outputs[t_name]
        aucs_tuple = []
        for i in range(len(out_probs)):
            auc = roc_auc_score(true_labels, out_probs[i])
            aucs_tuple.append((i, auc))
        sorted_auc = sorted(aucs_tuple, key=lambda x: x[1])
        median_model = sorted_auc[len(sorted_auc)//2]
        print('Median model (position,auc) = {}'.format(median_model))
        # Choose median model
        taggers_median[t_name] = out_probs[median_model[0]]
    return taggers_median


def get_median_bg_reject_model(true_labels, tagger_outputs):
    """Get median model sorted by background reject."""
    taggers_median = dict()
    # Initialize local variables
    point = 225
    base_tpr = np.linspace(0.05, 1, 476)
    for t_name in tagger_outputs:
        out_probs = tagger_outputs[t_name]
        out_scores = []
        for i in range(len(out_probs)):
            fpr, tpr, thresholds = roc_curve(
                true_labels,
                out_probs[i],
                pos_label=1,
                drop_intermediate=False
            )
            interp_fpr = interp(base_tpr, tpr, fpr)
            score = 1./interp_fpr[point]
            out_scores.append((i, score))
        sorted_scores = sorted(out_scores, key=lambda x: x[1])
        median_model = sorted_scores[len(sorted_scores)//2]
        print('Median model (position,auc) = {}'.format(median_model))
        # Choose median model
        taggers_median[t_name] = out_probs[median_model[0]]
    return taggers_median


def plot(
    tagger_results, input_truth_file, start_index, output_file, get_model
):
    """Create plot containing ROC corves for all tagger outputs. Saves the plot
    to the given output file.

    Parameters
    ----------
    tagger_results: list(dict)
        List containing information about the results for each tagger. Each
        item in the list is a dictionary with two elements: the unique tagger
        name ('name') and the path to the tagger result file ('file')
    input_truth_file: string
        Path to the file (labels.pkl) that contains the ground truth data
    start_index: int
        Start index for jet processing.
    output_file: string
        Path to the output file where the plot is stored
    get_model: function
        Function to get model from the tagger runs
    """
    # Load groundtruth labels. Keep only the labels that come after the start
    # index
    with open(input_truth_file, 'rb') as f:
        true_labels = np.asarray(pickle.load(f), dtype=int)
    true_labels = true_labels[start_index:]
    # Load tagger result data. Maintain list of tagger names in order of their
    # appearance in the result list and a dictionary that maps tagger names to
    # their output data.
    tagger_names = list()
    tagger_outputs = dict()
    for tagger in tagger_results:
        t_name = tagger['name']
        tagger_names.append(t_name)
        with open(tagger['file'], 'rb') as f:
            tagger_outputs[t_name] = pickle.load(f)
    tagger_median = get_model(true_labels, tagger_outputs)
    # Shortcut to tagger names and outputs
    base_tpr = np.linspace(0.05, 1, 476)
    # Define line styles for plot
    linestyles = ['-', '--', '-.', ':'] * 10
    for t_name in sorted(tagger_names):
        # Tagger output array
        t_out = tagger_median[t_name]
        fpr, tpr, thresholds = roc_curve(
            true_labels,
            t_out,
            pos_label=1,
            drop_intermediate=False
        )
        # Get error
        inv_fpr = interp(base_tpr, tpr, fpr)
        # Create plot and save to file (if filename is given)
        plt.rcParams['figure.figsize'] = 6, 6
        plt.plot(
            base_tpr,
            1./inv_fpr,
            alpha=1,
            label=str(t_name),
            linestyle=linestyles.pop(0)
        )
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel(r"Signal efficiency $\epsilon_S$", fontsize=15)
        plt.ylabel(r"Background rejection $\frac{1}{\epsilon_B}$", fontsize=15)
        plt.xlim([-0.01, 1.02])
        plt.ylim(4, 40000)
        plt.yscale("log")
        plt.legend(loc="best")
        plt.grid(which='both', axis='both', linestyle='--')
        plt.savefig(output_file)
