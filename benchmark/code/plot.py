# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def get_accuracy(outputs, labels):
    """Compute the accuracy for all tagger outputs."""
    # Flatten array
    labels = labels.ravel()
    # Round elements of the array to the nearest integer.
    outputs = np.rint(outputs)
    outputs = np.asarray(outputs)
    labels = np.asarray(labels)
    outputs = outputs.flatten()
    return np.sum(outputs == labels) / float(len(labels))


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


def plot(tagger_results, input_truth_file, tag_eff, output_file):
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
    tag_eff: float
    output_file: string
        Path to the output file where the plot is stored
    """
    # Load groundtruth labels
    with open(input_truth_file, 'rb') as f:
        true_labels = np.asarray(pickle.load(f), dtype=int)
    # Load tagger result data. Maintain list of tagger names in order of their
    # appearance in the result list and a dictionary that maps tagger names to
    # their output data.
    tagger_names = list()
    tagger_outputs = dict()
    for tagger in tagger_results:
        t_name = tagger['name']
        tagger_names.append(t_name)
        with open(tagger['file'], 'rb') as f:
            tagger_outputs[t_name] = np.asarray(pickle.load(f))
    tagger_median = get_median_model(true_labels, tagger_outputs)
    # Shortcut to tagger names and outputs
    # ???
    aucs = []
    errors = []
    accuracies = []
    base_tpr = np.linspace(0.05, 1, 476)
    # Get tag efficiency working point
    if tag_eff == 0.5:
        point = 225
    elif tag_eff == 0.3:
        point = 125
    elif tag_eff == 0.4:
        point = 175
    # Define line styles for plot
    linestyles = ['-', '--', '-.', ':'] * 10
    for t_name in tagger_names:
        # Tagger output array
        t_out = tagger_median[t_name]
        # Tagger accuracies, AUC, and ROC curve
        accuracies.append(get_accuracy(t_out, true_labels))
        aucs.append(roc_auc_score(true_labels, t_out))

        fpr, tpr, thresholds = roc_curve(
            true_labels,
            t_out,
            pos_label=1,
            drop_intermediate=False
        )
        # Get error
        inv_fpr = interp(base_tpr, tpr, fpr)
        len_bg = len(true_labels[true_labels == 0])
        bg_eff = inv_fpr[point]
        error_rel = np.sqrt(1./(bg_eff*len_bg))
        errors.append(error_rel*1./inv_fpr[point])
        # Create plot and save to file (if filename is given)
        plt.rcParams['figure.figsize'] = 9, 9
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
        plt.show()


if __name__ == '__main__':
    """This is a simple test function to call the plot function unsing the
    result from only a single tagger.
    """
    args = sys.argv[1:]
    if len(args) != 3:
        prog = os.path.basename(sys.argv[0])
        args = ['<tagger-result-file>', '<groundtruth-file>', '<output-file>']
        print('Usage: {} {}'.format(prog, ' '.join(args)))
        sys.exit(-1)
    results = list([{'name': 'TreeNiN', 'file': args[0]}])
    plot(
        tagger_results=results,
        input_truth_file=args[1],
        tag_eff=0.3,
        output_file=args[2]
    )
