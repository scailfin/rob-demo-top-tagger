# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

import os
import sys

from flowserv.service.postproc.client import Runs

import plot as plt


if __name__ == '__main__':
    """This is the main function to call the plot function unsing the result
    from a set of taggers. Selects the median model based on AUC.
    """
    args = sys.argv[1:]
    if len(args) != 3:
        prog = os.path.basename(sys.argv[0])
        args = ['<in_dir>', '<groundtruth-file>', '<output-dir>']
        print('Usage: {} {}'.format(prog, ' '.join(args)))
        sys.exit(-1)
    in_dir = args[0]
    input_truth_file = args[1]
    out_dir = args[2]
    # Read the submission information
    results = list()
    for run in Runs(in_dir):
        filename = run.get_file('results/yProbBest.pkl')
        results.append({'name': run.name, 'file': filename})
    # [{'name': 'TreeNiN', 'file': args[0]}]
    plt.plot(
        tagger_results=results,
        input_truth_file=input_truth_file,
        start_index=400000,
        output_file=os.path.join(out_dir, 'ROC-AUC.png'),
        get_model=plt.get_median_model
    )
