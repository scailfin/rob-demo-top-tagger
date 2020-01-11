import docker

from docker.errors import ContainerError, ImageNotFound, APIError

client = docker.from_env()

cmd = '''python code/preprocess-dataset.py
    data/test_jets.pkl
    data/preprocess/jet_image_trim_pt800-900_card.dat
    data/preprocess/transformer.pkl
    results/'''

try:
    r = client.containers.run(
        image='toptagger:1.0',
        command=cmd,
        volumes={
            '/home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/code': {'bind': '/code', 'mode': 'ro'},
            '/home/heiko/projects/scailfin/rob-demo-top-tagger/benchmark/data': {'bind': '/data', 'mode': 'ro'},
            '/home/heiko/projects/scailfin/rob-demo-top-tagger/results': {'bind': '/results', 'mode': 'rw'}
        }
    )
    print(r)
except (ContainerError, ImageNotFound, APIError) as ex:
    print(str(ex))
