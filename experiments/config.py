import numpy as np
from utils import RandomParameter, ArgPreprocessor, log_uniform, BooleanParameter

np.random.seed(4242)
servers = ["brown"]

experimentSuite = {
    'experiments': [{
        # Base Arguments
        'env': "MiniGrid-KeyCorridorS3R2-v0",
        'seed': [87],
        'batch-size': 1280,
        'log-interval': 10,
        'algo': 'ppo',

        # Training Arguments
        'frames': int(100500),
        'lr': 1e-4,
        'entropy-coef': 0.005,
        'recurrence': 10,
        'frames-per-proc': [40],
        'procs': 64,
        'epochs': 4,
        'reward-scale': 20.,
        'gae-lambda': 0.99,
        'max-grad-norm': 0.5,
        'clip-eps': 0.2,

        # Policy Model Arguments
        'arch': "cnn1",
        'image-dim': 128,
        'instr-dim': 0,
        'skill-dim': 32,
        'shared-image-embedding': True,
        'memory-dim': 128,

    }],
    'conditions': 'frames-per-proc',
    'num_hyperparameter_samples': 0,
    'label': '3101_Speed_Original'
}

own_username = "msmith"
repo_ssh_string = "git@github.com:mila-iqia/babyai.git"
project_dir = 'babyai'
project_name = f'speedcheck-{project_dir}-{own_username}'
script = "./scripts/train_rl.py"

# GPU Details
unique_id = 0
max_nr_processes = 3
upper_gpu_util = 50
