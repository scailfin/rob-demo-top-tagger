========================
Setup Python Environment
========================

This is a preliminary list of steps to setup a Python Anconda environment to run the Top Tagger dome.


.. code-block:: bash

    #
    # Setup conda environment
    #
    conda create -n toptagger python=3.7 pip numpy scipy scikit-learn


.. code-block:: bash

    #
    # Activate the toptagger environment
    #
    # Deactivate using: conda deactivate
    #
    conda activate toptagger


Install additional packages. Strangely, istalling PyTables using conda did not work due to module version conflicts.

.. code-block:: bash

    #
    # Install pytorch,
    #
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    #
    # Install PyTables for data extraction from HDF5 files
    #
    pip install tables
    #
    # Install pandas
    #
    conda install -c anaconda pandas


.. code-block:: bash

    #
    # Install fastjet
    #
    # Based on http://fastjet.fr/quickstart.html
    #
    cd /home/user/bin
    curl -O http://fastjet.fr/repo/fastjet-3.3.3.tar.gz
    tar zxvf fastjet-3.3.3.tar.gz
    cd fastjet-3.3.3/
    ./configure --enable-pyext --prefix=$PWD/../fastjet-install
    make
    make check
    make install


.. code-block:: bash

    #
    # Include reference to fastjet in environment paths.
    #
    # Based on https://github.com/JetsGame/GroomRL/issues/12
    #
    export LD_LIBRARY_PATH=/home/user/bin/fastjet-install/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/user/bin/fastjet-install/lib/python3.7/site-packages:$PYTHONPATH


Download and extract competition input data. For convenience, a version of the data is in included in the repository (downloaded on Jan. 9, 2020).

.. code-block:: bash

    python download-comp-data.py 400000 /tmp/test.h5 ../../data/jets/test_jets.pkl
