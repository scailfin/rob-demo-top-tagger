===========================================
Docker Container for Top Tagger Environment
===========================================

Build the container containing the Top Tagger environment:

.. code-block:: bash

    docker image build -t toptagger:1.0 .

Run workflow steps:

.. code-block:: bash

    docker container run \
        --rm \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code:/code \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data:/data \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/results:/results \
        --env LD_LIBRARY_PATH=/opt/fastjet-install/lib \
        --env PYTHONPATH=/opt/fastjet-install/lib/python3.7/site-packages \
        toptagger:1.0 \
        python code/preprocess-dataset.py \
            data/test_jets.pkl \
            data/preprocess/jet_image_trim_pt800-900_card.dat \
            data/preprocess/transformer.pkl \
            results/

    docker container run \
        --rm \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code:/code \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data:/data \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/results:/results \
        toptagger:1.0 \
        python code/evaluate-models.py \
            results/processed_test_jets.pkl \
            data/evaluate/ \
            results/

    docker container run \
        --rm \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code:/code \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data:/data \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/results:/results \
        toptagger:1.0 \
        python code/save-probabilities.py \
            data/evaluate/ \
            results/
