# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

import os, sys

import recnn.search as search

# Select start and finish model number for the scan. To evaluate the 9 models select Nstart=0 and Nfinish=9
Nstart=sys.argv[1]
Nfinish=sys.argv[2]

search.run()

os.chdir('/TreeNiN/code/recnn/')

# Load the trained weights and evaluate each model
os.system('python3 search_hyperparams.py --NrunStart='+str(Nstart)+' --NrunFinish='+str(Nfinish))

    output_file = os.path.join(output_dir, 'roc.pkl')
