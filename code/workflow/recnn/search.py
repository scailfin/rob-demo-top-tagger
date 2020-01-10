# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB).
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Peform hyperparemeters search"""

import logging
import os
import time

import recnn.utils as utils


# -- Main Function -------------------------------------------------------------

def run(params, static_data_dir, output_file):
    # Start pre-processing job. Main code block with the methods to load the
    # raw data, create and preprocess the trees.
    logging.info('Preprocessing jet trees ...')
    start_time = time.time()


    cmd_eval = "CUDA_VISIBLE_DEVICES={gpu} {python} evaluate.py --model_dir={model_dir} --data_dir={data_dir} --sample_name={sample_name} --jet_algorithm={algo} --architecture={architecture} --restore_file={restore_file}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=eval_data_dir,sample_name=sample_name, algo=algo, architecture=architecture, restore_file=restore_file)

    # Log runtime information
    exec_time = time.time()-start_time
    logging.info('Preprocessing time (minutes) = {}'.format(exec_time/60))
