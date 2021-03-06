# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Definition of names for files that are generated by a workflow run."""

# -- Preprocessing ------------------------------------------------------------
"""Tree file generated by the first dataset preprocessing step."""
RAW_TREE_FILE = 'tree_test_jets.pkl'
"""Result file of the dataset preprocessing step."""
PROCESSED_TREE_FILE = 'processed_test_jets.pkl'
"""Additional pre-processing input files."""
CARD_FILE = 'jet_image_trim_pt800-900_card.dat'
TRANSFORMER_FILE = 'transformer.pkl'

# -- Evaluate -----------------------------------------------------------------
"""Name of file containing run parameter dictionary."""
PARAMS_FILE = 'params.json'
"""Prefix for run directories."""
RUN_DIR_PREFIX = 'run_'
"""Result files for each run."""
METRICS_FILE = 'metrics_test.json'
ROC_FILE = 'roc.pkl'
Y_PROB_TRUE_FILE = 'yProbTrue.pkl'
"""Result files for run summaries."""
Y_PROB_BEST_FILE = 'yProbBest.pkl'
RESULT_FILE = 'results.json'


# -- Logging ------------------------------------------------------------------
"""Logfile for dataset preprocessing step."""
ANALYZE_LOG_FILE = 'analyze.log'
EVAL_LOG_FILE = 'evaluate.log'
PREPROC_LOG_FILE = 'preproc.log'
