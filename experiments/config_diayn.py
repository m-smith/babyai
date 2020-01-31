"""
Handles experiment configuration.

This file should not be used directly, but should be copied to
project_dir/experiments/config.py
"""
import random
import numpy as np
from utils import RandomParameter, ArgPreprocessor #, BinaryParameter
from utils import log_uniform

# The list of servers to check for available gpus
# servers = ['woma', 'savitar', 'whip', 'dgx1', 'brown', 'hydra',
#                          'gandalf', 'gimli', 'mulga', 'bilbo', 'legolas',
#                          'orion', 'tiger', 'taipan', 'sauron', 'gollum']
#random.shuffle(servers)
servers =  ['whip']
# We want consistent randomness for resuming lists of experiments
np.random.seed(4422)


"""
experimentSuite is a python dict with the following keys:
---------------------
experiments:
A list of dicts, each specifying configuration parameters for a single
experiment. It should contain a key "seed", specifying a list of seeds
to run for each experiment

conditions:
A string of comma separated keys from experiments for which the values are
expected to be lists, which will be expanded such that the key takes on each
value in the list.

label:
A unique string that identifies the experimentSuite

num_hyperparameter_samples:
(OPTIONAL - default:1) If a RandomParameter is specified for one or more values
of an experiment, this will be the number of unique experiments samled from the
joint distribution of all specified RandomParameters

branch:
(OPTIONAL - default:master) Which git branch to run the experiment from
---------------------
Special options for parameter values include:

RandomParameter(args_for_dist, distribution), which will use samples from
distribution(args_for_dist), to set this parameter, num_hyperparameter_samples
times.

BinaryParameter(), which indicates that this parameter should be expanded into
commands that either include this argument or not at all. Useful for
`action=store_true` parameters

ArgPreprocessor(args, preprocessing_fn), which will call
preprocessing_fn(**args), preprocessing_fn(*args), and preprocessing_fn(args),
setting the value of this parameter to the first successful result.
---------------------
The number of runs for each experiment in the suite will be:
#seeds * prod(#exp[c] for c in conditions) * num_hyperparameter_samples * (2 ** #BinaryParameter)
---------------------
"""

experimentSuite = {
    'experiments': [{
        # Base Arguments
        'env': ["MiniGrid-KeyCorridorS3R2-v0"], #], #, "MiniGrid-KeyCorridorS3R3-v0"],
        #  'env': ["MiniGrid-KeyCorridorS3R2-v0"], #"BabyAI-PutNextLocal-v0"], #, "MiniGrid-KeyCorridorS3R3-v0"],
        'seed': [1281, 1232],
        'batch-size': 1280,
        'log-interval': 10,
        'model-save-interval': int(1e6),
        'algo': 'ppo',

        # Training Arguments
        'frames': int(40e6),
        'lr': RandomParameter([1e-5, 1e-4], log_uniform),
        'entropy-coef': RandomParameter([0.005, 0.1], log_uniform),
        'recurrence': 10, # 20,
        'frames-per-proc': 40,
        'procs': 64,
        #'policy-epochs': 4, or epochs?
        'reward-scale': 1.,
        'gae-lambda': 0.99,
        'max-grad-norm': 0.5,
        'clip-eps': 0.2,
        'terminal-bootstrap-coef': [0,0.5,0.9,1],

        # Policy Model Arguments
        'arch': ['expert_filmcnn'],
        'image-dim': 128,
        'instr-dim': 0,
        'skill-dim': 32,
        'memory-dim': [128], # 256
        'dscr-image-dim': 64,
        'dscr-memory-dim': 0,
        'pseudo-reward-norm': True,

        # exploring
        'dscr-full-obs': False,
        'n-skills': 8,
        'dscr-lr': RandomParameter([1e-5, 1e-4], log_uniform),
        'dscr-epochs': [4],
        'dscr-batch-size': [640, 1280, 1920],
        'vod-states': 'all',
        'skill-length': 0,
        #

        # Validation Arguments
        # left for now

    }],
    'conditions': 'env,arch,dscr-epochs,memory-dim,terminal-bootstrap-coef,dscr-batch-size',
    'num_hyperparameter_samples': 2,
    'label': '1301_diayn_bootstrap_coef',
    'branch': 'transfer_explore',
}

#  'label': '1212_test_sweep' -- testing for bugs
#   'label': '1312_test_nstep'; vod-states=last, skill=6
#   'label': '1312_diayn_test'; vod-states=all, skill=length-all
#    'label': '1612_diayn_test',

own_username = "msmith"
repo_ssh_string = "git@github.com:oxwhirl/babyai-up.git"
project_dir = 'babyai-up'
project_name = f'{project_dir}-{own_username}'
script = 'scripts/pretrain_rl.py'

# GPU Details
unique_id = 3
max_nr_processes = 3
upper_gpu_util = 60
