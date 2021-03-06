"""
Handles experiment configuration.

This file should not be used directly, but should be copied to
project_dir/experiments/config.py
"""
import random
import numpy as np
from utils import RandomParameter, ArgPreprocessor, BinaryParameter
from utils import log_uniform

# The list of servers to check for available gpus
servers = ['woma', 'savitar', 'whip', 'dgx1', 'brown', 'hydra',
                          'gandalf', 'gimli', 'mulga', 'bilbo', 'legolas',
                          'orion', 'tiger', 'taipan', 'sauron', 'gollum']
random.shuffle(servers)
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
        'env': "BabyAI-1RoomS8-v0",
        # ["BabyAI-PickupDist-v0", "BabyAI-KeyCorridorS3R3-v0", "BabyAI-UnlockPickupDist-v0", "BabyAI-PutNextLocal-v0"],
        # 1RoomS8: Very Easy
        'seed': [987, 654, 321],
        'batch-size': 1280,
        'log-interval': 10,
        'algo': 'ppo',

        # Training Arguments
        'frames': int(1e9),
        'lr': RandomParameter([1e-5, 1e-3], log_uniform),
        'entropy-coef': RandomParameter([0.001, 0.01], log_uniform),
        'recurrence': 20,
        'frames-per-proc': 40,
        'procs': 64,
        'ppo-epochs': 4,

        # Model Arguments
        'image-dim': [1, 32],

        # Discriminator Arguments
        'n-skills': [4, 8, 16],
        'dscr-lr': RandomParameter([1e-5, 1e-3], log_uniform),
        'dscr-epochs': 4,
        'dscr-image-dim': [1, 32],
        'dscr-use-actions': BinaryParameter(),
        'dscr-action-dim': 32,

        # Validation Arguments
        # left for now

        # Pretraining Arguments
        'reward-scale': 1,
        'gae-lambda': 0.99,
        'max-grad-norm': 0.5,
        'clip-eps': 0.2,

    }],
    'conditions': 'n-skills,dscr-image-dim,image-dim',
    'num_hyperparameter_samples': 4,
    'label': '1308_dscr_debug_img_dim_2'
}

own_username = "msmith"
repo_ssh_string = "git@github.com:oxwhirl/babyai-up.git"
project_dir = 'babyai-up'
project_name = 'babyai-up'

# GPU Details
unique_id = 0
max_nr_processes = 3
upper_gpu_util = 50
