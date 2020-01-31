import numpy as np
from utils import RandomParameter, ArgPreprocessor, log_uniform, BooleanParameter

np.random.seed(4242)
servers = ["brown"]

experimentSuite = {
    'experiments': [{
        # Base Arguments
        'env': "MiniGrid-DenseGoal-v0",#"MiniGrid-KeyCorridorS3R2-v0",
        'seed': [87],
        'batch-size': 1280,
        'log-interval': 10,
        'algo': 'ppo',

        # Training Arguments
        'frames': int(200500),
        'rl-frames': int(100000),
        'transfer-frames': int(200000),
        'lr': 1e-4,
        'entropy-coef': 0.005,
        'recurrence': 10,
        'frames-per-proc': 40,
        'procs': 64,
        'policy-epochs': 4,
        'reward-scale': 20.,
        'pseudo-reward-scale':1,
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

        # Discriminator Model Arguments
        'dscr-image-dim': 64,
        'dscr-memory-dim': 0,
        'dscr-lr': 1e-4,
        'dscr-epochs': [4],
        'dscr-batch-size': [1280], # possibly too high for classification batch size
        'dscr-full-obs': False,

        # Exploration Policy Model Arguments
        'exploration-arch': "expert_filmcnn",
        "fix-dscr-embedding": False,
        "goal-obs-dim": 3,

        # Diversity Arguments
        'n-skills': 8,
        'vod-states': 'all',
        'pseudo-reward-norm' : True,
        'skill-length': 0,
        'terminal-bootstrap-coef': 0

    }],
    'conditions': 'dscr-epochs,dscr-batch-size',
    'num_hyperparameter_samples': 0,
    'label': '2101_Profiling_2',
    'branch': "transfer_explore"
}

own_username = "msmith"
repo_ssh_string = "git@github.com:oxwhirl/babyai-up.git"
project_dir = 'babyai-up'
project_name = f'profiling-fix2-{project_dir}-{own_username}'
script = " -m cProfile -o profile.stat ./scripts/train_transfer.py"

# GPU Details
unique_id = 0
max_nr_processes = 3
upper_gpu_util = 50
