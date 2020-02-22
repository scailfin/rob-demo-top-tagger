===========================================
Docker Container for Top Tagger Environment
===========================================

Build the container containing the Top Tagger environment:

.. code-block:: bash

    docker image build -t toptaggerdemo:1.1 .


Push container image to DockerHub.

.. code-block:: bash

    docker image tag toptaggerdemo:1.1 heikomueller/toptaggerdemo:1.1
    docker image push heikomueller/toptaggerdemo:1.1


Run workflow steps:

.. code-block:: bash

    docker container run \
        --rm \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code:/code \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data:/data \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/results:/results \
        heikomueller/toptaggerdemo:1.1 \
        python code/preprocess-dataset.py \
            data/test_jets.pkl \
            data/preprocess/ \
            results/

    docker container run \
        --rm \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code:/code \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data:/data \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/results:/results \
        heikomueller/toptaggerdemo:1.1 \
        python code/evaluate-models.py \
            results/processed_test_jets.pkl \
            data/evaluate/ \
            results/

    docker container run \
        --rm \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code:/code \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data:/data \
        -v /home/heiko/projects/scailfin/rob-demo-top-tagger/results:/results \
        heikomueller/toptaggerdemo:1.1 \
        python code/compute-score.py \
            data/evaluate/ \
            results/
