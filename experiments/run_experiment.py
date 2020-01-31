import argparse

from server_interface import startExperimentSet, inspect_gpus
from config import experimentSuite
import config
from utils import ArgPreprocessor, RandomParameter, BooleanParameter

parser = argparse.ArgumentParser(description="Runs experiments remotely")
parser.add_argument("--start-index", help="The first experiment to run",
                    type=int, default=0)


def experiment_preprocessing(experimentSuite):
    """
    Handles expansion of specific types of parameter search arguments
    """
    experiment_list = []
    for exp in experimentSuite["experiments"]:
        to_sample = []
        new_exps = [{**exp}]
        for key, value in exp.items():
            if isinstance(value, BooleanParameter):
                additional_exps = []
                for new_exp in new_exps:
                    additional_exps.append({**new_exp, **{key: ""}})
                    new_exp.pop(key)
                new_exps += additional_exps
            elif isinstance(value, RandomParameter):
                to_sample.append((key, value))
            elif isinstance(value, ArgPreprocessor):
                for new_exp in new_exps:
                    new_exp[key] = value()

        exps_with_samples = []
        if to_sample:
            n_samples = experimentSuite.get("num_hyperparameter_samples", 1)
            for _ in range(n_samples):
                sampled_values = {}
                for key, dist in to_sample:
                    sampled_values[key] = dist()
                for new_exp in new_exps:
                    exps_with_samples.append({**new_exp, **sampled_values})
        else:
            exps_with_samples = new_exps

        experiment_list += exps_with_samples
    return experiment_list


def expandListsInExperiments(experimentSuite, fields_to_expand, conditions, label):
    experiments = experiment_preprocessing(experimentSuite)
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


experimentSuite = expandListsInExperiments(experimentSuite,
                                           experimentSuite['conditions'].split(",") + ["seed"],
                                           conditions=experimentSuite['conditions'],
                                           label=experimentSuite['label'])



if __name__ == '__main__':
    print("Number of experiments:", len(experimentSuite['experiments']))
    import pprint
    pprint.pprint(experimentSuite['experiments'])

    args = parser.parse_args()
    ### Cut into smaller pieces
    experimentSuite['experiments'] = experimentSuite['experiments'][args.start_index:]
    #%%

    print("Looking for {} GPUs".format(len(experimentSuite['experiments'])))

    free_gpus = inspect_gpus(
        own_username=config.own_username,
        verbose=True,
        servers=config.servers,
        needed_gpus=len(experimentSuite['experiments']),
        allow_lightly_used_gpus=True,
        share_with=['all'],
        upper_gpu_util_threshold=config.upper_gpu_util,
        upper_memory_threshold=5500,
        max_nr_processes=config.max_nr_processes,
        average=3)

    running_experiments, ids = startExperimentSet(
        experiments=experimentSuite['experiments'],
        script=config.script,
        free_gpus=free_gpus,
        project_dir=config.project_dir,
        project_name=config.project_name,
        repo_ssh_string=config.repo_ssh_string,
        sleep_time=10,
        overrule_repo_check=True,
        rebuild_docker=True,
        update_repo=True,
        unique_id=config.unique_id,
        branch=experimentSuite.get("branch", None))
