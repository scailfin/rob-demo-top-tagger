# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Demo evaluation file for algorithm TreeNiN++."""

import argparse
import evaluate_models as eval


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
        default=5,
        help='End model number.'
    )
    parser.add_argument('tree_file', help='Preprocessed tree dataset.')
    parser.add_argument('data_dir', help='Base directory for run parameters.')
    parser.add_argument('output_dir', help='Base directory for run results.')
    eval.main(args=parser.parse_args())
