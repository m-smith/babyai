from server_interface import startExperimentSet, inspect_gpus
from server_interface import createCommand, updateRepo

from plumbum import SshMachine

#servers = [ 'woma', 'savitar', 'whip', 'dgx1', 'brown', 'sauron', 'gollum', 'gandalf', 'gimli', 'mulga', 'bilbo', 'legolas', 'orion', 'hydra','tiger', 'taipan']
servers = ['sauron', 'brown', 'woma', 'gollum', 'whip', 'gimli', 'orion', 'savitar', 'dgx1']


def expandListsInExperiments(experimentSuite, fields_to_expand, conditions, label):
    experiments = experimentSuite['experiments']
    while fields_to_expand:
        new_experiments = []
        field_to_expand = fields_to_expand.pop()

        for exp in experiments:
            new_exps = []
            for value in exp[field_to_expand]:
                new_exp = exp.copy()
                new_exp[field_to_expand] = value
                new_exp['conditions'] = conditions
                new_exp['label'] = label
                new_exps.append(new_exp)
            new_experiments += new_exps
        experiments = new_experiments
        experimentSuite['experiments'] = new_experiments
    return experimentSuite

#%%

experimentSuite = {
    'experiments': [{

        # Base Arguments
        'env': "BabyAI-UnlockPickupDist-v0",
        # ["BabyAI-PickupDist-v0", "BabyAI-KeyCorridorS3R3-v0", "BabyAI-UnlockPickupDist-v0", "BabyAI-PutNextLocal-v0"],
        'seed': [123, 234, 345],
        'batch-size': 1280,
        'log-interval': 10,
        'algo' : 'ppo',

        # Training Arguments
        'frames': int(1e9),
        'lr': 1e-4,
        'entropy-coef': [0.01, 0.03, 0.05],#[0.01, 0.03, 0.05],
        'recurrence': 20,
        'frames-per-proc' : 40,
        'procs': 64,
        'ppo-epochs': 4,

        # Model Arguments
        'image-dim': 128,

        # Discriminator Arguments
        'n-skills': [4, 8, 16],#[4, 8, 16],
        'dscr-lr': 1e-4,
        'dscr-epochs': 4,
        'dscr-image-dim': 64,

        # Validation Arguments
        # left for now

        # Pretraining Arguments
        'reward-scale': 20,
        'gae-lambda': 0.99,
        'max-grad-norm': 0.5,
        'clip-eps': 0.2,

    }],
    'conditions': 'n-skills,entropy-coef',
    'label': '0108_entropy_skills_sweep_unlock'}

experimentSuite = expandListsInExperiments(experimentSuite,
                                           ['seed']+experimentSuite['conditions'].split(","),
                                           conditions=experimentSuite['conditions'],
                                           label=experimentSuite['label'])

## Set unique id for runs
unique_id = 0
max_nr_processes = 3
upper_gpu_util = 50

### Cut into smaller pieces
# experimentSuite['experiments'] = experimentSuite['experiments'][7:]

print("Number experiments:", len(experimentSuite['experiments']))
import pprint
pprint.pprint(experimentSuite['experiments'])

#%%

print("Looking for {} GPUs".format(len(experimentSuite['experiments'])))
free_gpus = inspect_gpus(
    own_username='jelina',
    verbose=True,
    servers=servers,
    needed_gpus=len(experimentSuite['experiments']),
    allow_lightly_used_gpus=True,
    share_with=['all'],
    upper_gpu_util_threshold=upper_gpu_util,
    upper_memory_threshold=5500,
    max_nr_processes=max_nr_processes,
    average=3)
len(free_gpus)

running_experiments, ids = startExperimentSet(
    experiments=experimentSuite['experiments'],
    free_gpus=free_gpus,
    project_dir='babyai-up',
    project_name='babyai-up',
    repo_ssh_string='git@github.com:oxwhirl/babyai-up.git',
    sleep_time=10,
    overrule_repo_check=True,
    rebuild_docker=True,
    update_repo=True,
    unique_id=unique_id)
