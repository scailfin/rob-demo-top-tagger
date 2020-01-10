# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB).
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Train the model"""
"""This is from the pytorch_shuffle dir"""

import logging
import numpy as np
import os
import pickle
import sys
import time

import recnn.model.data_loader as dl
import recnn.utils as utils

from recnn.model import preprocess


# -- Main preprocessing function -----------------------------------------------

def run(params, input_tree_file, transformer_file, output_file):
    # Adjust parameters for preprocessing job
    params.number_of_labels_types=1
    params.learning_rate=2e-3
    params.decay=0.9
    params.batch_size=400
    params.save_summary_steps=400
    params.num_epochs=40
    params.hidden=50
    params.features=7
    params.myN_jets=1200000
    params.nrun_start=0
    params.nrun_finish=1
    # Start pre-processing job. Main code block with the methods to load the
    # raw data, create and preprocess the trees.
    logging.info('Preprocessing jet trees ...')
    start_time = time.time()
    # Load pre-processed top tagger reference test tree data
    data_loader = dl.DataLoader
    logging.info('Loading toptag_reference_dataset {}'.format(input_tree_file))
    with open(input_tree_file, 'rb') as f:
        tt_ref_test_data = pickle.load(f,encoding='latin-1')
    # Preprocess
    tt_ref_x = list()
    for jet in np.asarray([x for (x,y) in tt_ref_test_data]):
        cont = preprocess.rewrite_content(jet)
        samples = preprocess.extract_nyu_samples(preprocess.permute_by_pt(cont))
        tt_ref_x.append(samples)
    tt_ref_y = np.asarray([y for (x,y) in tt_ref_test_data])
    # Load transformer
    logging.info('Loading transformer {}'.format(transformer_file))
    with open(transformer_file, 'rb') as f:
        transformer = pickle.load(f)
    # Scale features using the training set transformer
    tt_ref_x = data_loader.transform_features(transformer, tt_ref_x)
    # Save trees
    logging.info('Write trees to {}'.format(output_file))
    with open(output_file, 'wb') as f:
        pickle.dump(zip(tt_ref_x, tt_ref_y), f)
    # Log runtime information
    exec_time = time.time()-start_time
    logging.info('Preprocessing time (minutes) = {}'.format(exec_time/60))
