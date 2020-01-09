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

def run(config_dir, static_data_dir, run_dir, nrun_start, nrun_finish):
    # Load the "reference" parameters from parent_dir json file
    json_path = os.path.join(config_dir, 'template_params.json')
    params = utils.Params(json_path)
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
    params.nrun_start=nrun_start
    params.nrun_finish=nrun_finish
    # Start pre-processing job. Main code block with the methods to load the
    # raw data, create and preprocess the trees.
    logging.info('Preprocessing jet trees ...')
    start_time = time.time()

    cmd_eval = "CUDA_VISIBLE_DEVICES={gpu} {python} evaluate.py --model_dir={model_dir} --data_dir={data_dir} --sample_name={sample_name} --jet_algorithm={algo} --architecture={architecture} --restore_file={restore_file}".format(gpu=GPU, python=PYTHON, model_dir=model_dir, data_dir=eval_data_dir,sample_name=sample_name, algo=algo, architecture=architecture, restore_file=restore_file)

    # Log runtime information
    exec_time = time.time()-start_time
    logging.info('Preprocessing time (minutes) = {}'.format(exec_time/60))
