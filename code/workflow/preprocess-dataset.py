# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Default data pre-processing step for the Top Tagger competition workflow.
This step creates the initial input trees from the test data and applies steps
to get the initial input data features.

The dataset preprocessor performs two main steps:

- Creates binary trees with the clustering history of the jets and outputs
  a dictionary for each jet that contains the root_id, tree, content
  (constituents 4-momentum vectors), mass, pT, energy, eta and phi values.
- Preprocessing is applied. The initial 7 features are: p, eta, phi, E, E/JetE,
  pT, theta.

This is an adoped version of the code in:
https://github.com/SebastianMacaluso/TreeNiN
"""

import logging
import os
import sys

import dataset.preprocess as pf
import recnn.preprocess as recnn


# -- Main Function -------------------------------------------------------------

if __name__ == '__main__':
    """Create the input trees. Load and recluster the jet constituents. Create
    binary trees with the clustering history of the jets and output a dictionary
    for each jet that contains the root_id, tree, content (constituents
    4-momentum vectors), mass, pT, energy, eta and phi values (also charge,
    muon ID, etc depending on the information contained in the dataset).

    The preprocessing step expects four command line parameters:
    - dataset-file (test_jets.pkl)
    - card-file (jet_image_trim_pt800-900_card.dat)
    - transformer-file (transformer.pkl)
    - output-dir

    Creates two files in the given output directory:

    - tree_test_jets.pkl
    - processed_test_jets.pkl
    """
    # Get the five command line parameters that define the input and output
    # files for the pre-processing step.
    args = sys.argv[1:]
    if len(args) != 4:
        msg = 'Usage: {} <dataset-file> <card-file> <transformer-file> <output-dir>'
        print(msg.format(sys.argv[0]))
        sys.exit(-1)
    input_jets_file = args[0]
    card_file = args[1]
    transformer_file = args[2]
    out_dir = args[3]
    # Ensure that the output directory for preprocessing output file exists
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Initialize the logger
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)
    #
    # -- Step 1 ----------------------------------------------------------------
    # Run jets preprocessing task that uses FastJet to create the tree test
    # jets.
    #
    tree_file = os.path.join(out_dir, 'tree_test_jets.pkl')
    pf.run(
        card_file=card_file,
        input_jets_file=input_jets_file,
        out_file=tree_file
    )
    #
    # -- Step 2 ----------------------------------------------------------------
    # Apply preprocessing: get the initial 7 features: p, eta, phi, E, E/JetE,
    # pT, theta. Apply RobustScaler
    #
    recnn.run(
        tree_file=tree_file,
        transformer_file=transformer_file,
        output_file=os.path.join(out_dir, 'processed_test_jets.pkl')
    )
