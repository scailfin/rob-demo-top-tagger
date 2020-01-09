# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB).
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Default data pre-processing step for the Top Tagger competition workflow.
This step creates the initial input trees from the test data and applies steps
to get the initial input data features.

This is an adoped version of the code in:
https://github.com/SebastianMacaluso/TreeNiN
"""

import logging
import os
import sys

import dataset.preprocess as pf
import recnn.preprocess as recnn


# -- Main Function -------------------------------------------------------------

# Create the input trees.
# Load and recluster the jet constituents. Create binary trees with the clustering history of the jets and output a dictionary for each jet that contains the root_id, tree, content (constituents 4-momentum vectors), mass, pT, energy, eta and phi values (also charge, muon ID, etc depending on the information contained in the dataset)
# FastJet needs python2.7
if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    # Ensure that the output directory for preprocessing results exists
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Expected sub-folders in the given data directory
    jets_dir = os.path.join(data_dir, 'jets')
    conf_dir = os.path.join(data_dir, 'config')
    # Initialize the logger
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)
    # Run jets preprocessing task that uses FastJet to create the tree test
    # jets.
    pf.run(
        card_file=os.path.join(jets_dir, 'jet_image_trim_pt800-900_card.dat'),
        sample_type='test',
        dir_subjets=jets_dir,
        out_file=os.path.join(out_dir, 'tree_test_jets.pkl')
    )
    # Apply preprocessing: get the initial 7 features: p, eta, phi, E, E/JetE,
    # pT, theta. Apply RobustScaler
    recnn.run(config_dir=conf_dir, static_data_dir=jets_dir, run_dir=out_dir)
