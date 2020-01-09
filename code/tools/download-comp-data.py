# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB).
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Download Top Tagger competition test dataset from desycloud.desy.de.

This is an adoped version of the ReadData.py code in:
https://github.com/SebastianMacaluso/TreeNiN

 usage: download-comp-data.py [-h] [-o OVERWRITE] [-c CLEANUP]
                             n_start h5_file array_file

Download and extract competition test dataset.

positional arguments:
  n_start               Start record number.
  h5_file               Downloaded test data file.
  array_file            Extracted jet array output file.

optional arguments:
  -h, --help            show this help message and exit
  -o OVERWRITE, --overwrite OVERWRITE
                        Ovewrite existing h5 file
  -c CLEANUP, --cleanup CLEANUP
                        Delete downloaded h5 file after data extraction.
"""

import argparse
import logging
import numpy as np
import os
import pandas
import pickle
import urllib.request
import sys


"""Test data download Url."""
DOWNLOAD_URL = 'https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=test.h5'


# -- Helper Functions ----------------------------------------------------------

def h5_to_npy(filename, n_start):
    """This function reads the h5 files and saves the jets in numpy arrays. We
    only save the non-zero 4-vector entries.

    Parameters
    ----------
    filename: string
        Path to the local test.h5 input file
    n_start: int
        Start record number

    Returns
    -------
    np.array
    """
    file = pandas.HDFStore(filename)
    jets = np.array(file.select("table", start=n_start, stop=None))
    # This way I'm getting the 1st 199 constituents. jets[:,800:804] is the
    # constituent 200. jets[:,804] has a label=0 for train, 1 for test, 2 for
    # val. jets[:,805] has the label sg/bg
    jets2 = jets[:,0:800].reshape((len(jets), 200, 4))
    labels = jets[:,805:806]
    npy_jets = []
    for i in range(len(jets2)):
        # Get the index of non-zero entries
        nonzero_entries = jets2[i][~np.all(jets2[i] == 0, axis=1)]
        npy_jets.append([nonzero_entries, 0 if labels[i] == 0 else 1])
    # Close input file and return array
    file.close()
    return npy_jets


# -- Main Function -------------------------------------------------------------

if __name__ == '__main__':
    # Get record start number and output file names for the downloaded test
    # data and the extracted jet arrays. The optional OVERWRITE and CLEANUP
    # parameter are used to control behavior regarding the test.h5 file
    # download and storage.
    parser = argparse.ArgumentParser(
        description='Download and extract competition test dataset.'
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Ovewrite existing h5 file'
    )
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        '-c', '--cleanup',
        action='store_true',
        help='Delete downloaded h5 file after data extraction.'
    )
    parser.set_defaults(cleanup=False)
    parser.add_argument('n_start', type=int, help='Start record number.')
    parser.add_argument('h5_file', help='Downloaded test data file.')
    parser.add_argument('array_file', help='Extracted jet array output file.')
    args = parser.parse_args()
    print(args)
    # Initialize the logger
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.DEBUG)
    # Download the h5 test file if it does not exist or if the OVERWRITE flag
    # is True
    if not os.path.isfile(args.h5_file) or args.overwrite:
        logging.info('Download test.h5 to {} ...'.format(args.h5_file))
        urllib.request.urlretrieve(DOWNLOAD_URL, args.h5_file)
    else:
        logging.info('Skip download of test.h5')
    # Extract jet arrays and write to array outout file
    logging.info('Extracting jet constituents ...')
    test_data = h5_to_npy(args.h5_file, n_start=args.n_start)
    with open(args.array_file, 'wb') as f:
        pickle.dump(test_data, f, protocol=2)
    # Optional cleanup step
    if args.cleanup:
        logging.info('Remove {}'.format(args.h5_file))
        os.remove(args.h5_file)
