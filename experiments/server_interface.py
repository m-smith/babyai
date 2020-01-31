#!/usr/bin/env python
# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

"""
Functions to check servers for free GPUs and start experiment on those GPUs.
See example file for usage.
Call this file from command line to search servers for free GPUs and print the result.
"""
import json
import re
import subprocess
import re
import argparse
from plumbum import SshMachine
from plumbum.cmd import sed, awk, git
import plumbum
import csv
from plumbum import colors
import time
from git import Repo

parser = argparse.ArgumentParser(description='Check GPU usage')
parser.add_argument('--verbose', action='store_true',
                    default=False)
parser.add_argument("--max_nr_processes", type=int, default=3)
parser.add_argument("--average", type=int, default=3)
parser.add_argument('--clearall', action='store_true')


color_free = colors.green
color_me = colors.yellow
color_other = colors.blue
color_other_light = colors.sky_blue2


def inspect_gpus(servers,
                 own_username='jelina',
                 verbose=True,
                 needed_gpus=-1,
                 memory_threshold=1200,
                 gpu_util_threshold=5,
                 allow_lightly_used_gpus=True,
                 share_with=['all'],
                 max_nr_processes=6,
                 upper_memory_threshold=5500,
                 upper_gpu_util_threshold=60,
                 average=3,
                 get_all=False):
    """
    Scan servers for free GPUs, print availability and return a list of free GPUs that can used to
    start jobs on them.

    Requirements:
        ~/.ssh/config needs to be set up so that connecting via `ssh <server>` works. Fos OSX,
        an entry can look like this:

        Host mulga
            User maxigl
            HostName mulga.cs.ox.ac.uk
            BatchMode yes
            ForwardAgent yes
            StrictHostKeyChecking no
            AddKeysToAgent yes
            UseKeychain yes
            IdentityFile ~/.ssh/id_rsa

    Args:
        verbose (bool):           If True, also print who is using the GPUs
        server (list of strings): List of servers to scan


        memory_threshold (int):
        gpu_util_threshold (int): When used memory < lower_memory_threshold and
                                  GPU utilisation < lower_gpu_util_threshold,
                                  then the GPU is regarded as free.

        allow_lightly_used_gpus (bool):
        share_with (list of strings):
        upper_memory_threshold (int):
        upper_gpu_util_threshold (int): If `allow_lightly_used_gpus=True` and memory and gpu
                                        utilisation are under the upper thresholds and there
                                        is so far only one process executed on that GPU who's
                                        user is in in the list `share_with`, then the GPU will
                                        be added to the list of GPUs that can be used to start jobs.

    Return:
        free_gpus: List of dictionaries, each containing the following keys:
                   'server': Name of the server
                   'gpu_nr': Number of the free GPU
                   'double': Whether someone is already using that GPU but it's still considered
                             usuable (see `allow_lightly_used_gpus`)


    """
    print((color_free | "Free" + " | ") +
          (color_me | "Own" + " | ") +
          (color_other | "Other" + " | ") +
          (color_other_light | "Other (light)"))

    all_free_gpus = []
    server_id = 0

    while ((needed_gpus < 0 and server_id < len(servers)) or
            len(all_free_gpus) < needed_gpus) or get_all:
        try:
            server = servers[server_id]
        except:
            break
        server_id += 1
        print("{:7}: ".format(server), end='')
        try:
            remote = SshMachine(server)
        except plumbum.machines.session.SSHCommsError:
            print("ssh fail - server not in .ssh/config? See doctring of this function.")
            continue
        r_smi = remote["nvidia_smi"]
        r_ps = remote["ps"]
        averaged_gpu_data = []
        for avg_idx in range(average):
            fieldnames = ['index', 'gpu_uuid', 'memory.total', 'memory.used',
                        'utilization.gpu', 'gpu_name']
            output = r_smi("--query-gpu=" + ",".join(fieldnames),
                        "--format=csv,noheader,nounits").replace(" ", "")

            gpu_data = []
            for line in output.splitlines():
                gpu_data.append(dict([(name, int(x)) if x.strip().isdigit() else (name, x)
                                for x, name in zip(line.split(","), fieldnames)]))
            if avg_idx == 0:
                averaged_gpu_data = gpu_data
                for gpu_idx in range(len(averaged_gpu_data)):
                    averaged_gpu_data[gpu_idx]['utilization.gpu'] /= average
                    averaged_gpu_data[gpu_idx]['memory.used'] /= average
            else:
                for gpu_idx, data in enumerate(gpu_data):
                    averaged_gpu_data[gpu_idx]['utilization.gpu'] += data['utilization.gpu'] / average
                    averaged_gpu_data[gpu_idx]['memory.used'] += data['memory.used'] / average
            time.sleep(1)

        gpu_data = averaged_gpu_data


        # Find processes and users
        for data in gpu_data:
            data['nr_processes'] = 0
            data['users'] = []

        output = r_smi("--query-compute-apps=pid,gpu_uuid",
                       "--format=csv,noheader,nounits").replace(" ", "")

        gpu_processes = []
        for line in output.splitlines():
            gpu_processes.append([int(x) if x.strip().isdigit() else x for x in line.split(",")])

        for process in gpu_processes:
            pid = process[0]
            user = (r_ps['-u', '-p'] | sed['-n', '2p'] | awk['{{print $1}}'])(pid)
            serial = process[1]
            for data in gpu_data:
                if data['gpu_uuid'] == serial:
                    data['users'].append(user.strip())
                    data['nr_processes'] += 1

        gpu_numbers = []
        gpu_status = []
        free_gpus = []

        for data in gpu_data:
            status = "\t"+str(data['index']) + ": "
            # availability conditions: < 50MB and <5% utilisation ?

            # Is it free?
            if (data['memory.used'] < memory_threshold and
                data['utilization.gpu'] < gpu_util_threshold):

                status += "free"
                status = color_free | status
                gpu_numbers.append(color_free | str(data['index']))
                free_gpus.append({'server': server,
                                  'gpu_nr': data['index'],
                                  'occupation': 0})
                                  # 'session': getSession(data['index'])})
            else:
                status += "in use - {:4}% gpu - {:5}% mem - {}".format(
                    str(data['utilization.gpu'])[:4],
                    str(data['memory.used']/data['memory.total'])[:4],
                    str(data['users']))

                if 'all' in share_with:
                    share = True
                else:
                    share = data['users'][0] in share_with


                if (allow_lightly_used_gpus and
                    data['memory.used'] < upper_memory_threshold and
                    data['utilization.gpu'] < upper_gpu_util_threshold and
                    data['nr_processes'] < max_nr_processes and
                    share):

                    free_gpus.append({'server': server,
                                      'gpu_nr': data['index'],
                                      'occupation': data['nr_processes']})
                                      # 'session': getSession(data['index'] + 10)})
                    gpu_numbers.append(color_other_light | str(data['index']))
                    status = color_other_light | status
                else:
                    gpu_numbers.append(color_other | str(data['index']))
                    status = color_other | status
                # elif (own_username in data['users']):
                #     gpu_numbers[-1] = color_me | str(data['index'])

            gpu_status.append(status)

        all_free_gpus += free_gpus
        print(" ".join(gpu_numbers) + " | {} free | {} total".format(
            len(free_gpus),
            len(all_free_gpus)))

        if verbose:
            print("\t [{} - {} GB]".format(gpu_data[0]['gpu_name'],
                                           gpu_data[0]['memory.total'] // 1000))
            for s in gpu_status:
                print(s)

        remote.close()
    return all_free_gpus


def clearServers(servers, project_dir, repo_ssh_string, branch=None):
    # MS: I think I may have broken this with the whole branch thing
    for server in servers:
        print(server)
        try:
            remote = SshMachine(server)
        except plumbum.machines.session.SSHCommsError:
            print("ssh fail - server not in .ssh/config? See doctring of this function.")
            continue
        updateRepo(remote, project_dir, repo_ssh_string, branch=branch)
        r_clearDocker = remote[remote.cwd / project_dir / "docker/clear.sh"]
        output = r_clearDocker()
        print(output)


def startExperimentSet(experiments,
                       script,
                       free_gpus,
                       project_dir,
                       project_name,
                       repo_ssh_string,
                       sleep_time=10,
                       overrule_repo_check=False,
                       rebuild_docker=False,
                       update_repo=True,
                       repo_path="../",
                       unique_id=0,
                       branch=None
                       ):
    """
    Requirements:
        ~/<project_dir>/docker/run.sh must exist and take two arguments
             1. The GPU to be used.
             2. The name of the docker contain to be created
             3. The command to be executed
        ~/<project_dir>/docker/build.sh must exist
        (That means that the docker script provided in pool/documentation needs to be adapted slightly)

        Also, the project directory on the server should be located in the home directory.

    Args
        experiments: List or dictionaries. Each list item corresponds to one experiment to be started.
                     The key-value pairs are the command line configs added to the call. (see example below)
        project_dir: Name of the project directory (assumed to be located in home directory).
        project_name: Name of the project. Is used to create the container name.
        repo_ssh_string: Github ssh string for repo. Used to clone or pull latest updates.
        sleep_time (int): Time in seconds to wait after starting the last experiment before checking whether all are running.
        overrule_repo_check: If true, don't check whether current repo is dirty.
        rebuild_docker: If true, rebuild docker container before starting experiments.
        repo_path: Path to local git repository.

    Return
        running_experiments: List of started experiments
        ids: List of sacred ids of started experiments
    """
    print("Starting experiments...")
    if overrule_repo_check:
        print("WARNING: No git check is performed!")
    else:
        repo = Repo(repo_path)
        assert not repo.is_dirty()

    running_experiments = []
    for gpu, exp in zip(free_gpus[:len(experiments)], experiments):
        print(exp)
        command = createCommand(script, exp, gpu)
        print(gpu, command)
        container_name = getSession(project_name, gpu, unique_id)
        startExperiment(gpu, container_name, command, project_dir, repo_ssh_string,
                        update_repo=update_repo, rebuild_docker=rebuild_docker,
                        branch=branch)
        running_experiments.append({'gpu': gpu, 'config':exp})

    print('Waiting {} seconds...'.format(sleep_time))
    print(running_experiments)
    time.sleep(sleep_time)

    print('Checking whether experiments are running...')
    ids = []
    for started_experiment in running_experiments:
        id_ = inspect_container(project_name, started_experiment['gpu'], unique_id)
        print(id_, started_experiment)
        ids.append(id_)

    return running_experiments, ids


###################
# Private Methods #
###################


def createCommand(script, exp, gpu):
    command = f"python {script} " + " ".join(["--{} {}".format(key, exp[key]) for key in exp])
    #command = "python ./scrips/pretrain_rl.py with " + " ".join(["{}={}".format(key, exp[key]) for key in exp])
    command += " --server {} --gpu-id {}".format(gpu['server'], gpu['gpu_nr'])
    return command


def inspect_container(project_name, gpu, unique_id):
    remote = SshMachine(gpu['server'])
    session = getSession(project_name, gpu, unique_id)

    r_docker = remote['docker']
    inspect = r_docker('inspect', session)
    data = json.loads(inspect)
    # print(data[0]['State']['Running'])
    if data[0]['State']['Running']:
        logs = r_docker('logs', session)
        m = re.search('Started run with ID "(\d+)"', logs)
        id_ = int(m.group(1))
        return id_
    else:
        raise Exception("Container {} not running".format(unique_id))


def startExperiment(gpu, session, command, project_dir, repo_ssh_string,
                    update_repo=True, rebuild_docker=False, branch=None):
    """
    Helper function to start an experiment remotely.

    Requirements:
        <project_dir>/docker/run.sh must exist and take two arguments
             1. The name of the docker contain to be created
             2. The command to be executed
        <project_dir>/docker/build.sh must exist

        Also, the project directory should be located in the home directory.

    Args:
        gpu (int): Id of the GPU
        session (string): Name of the container to be created
        command (string): Command to be exectued
        project_dir (string): Name of the project directory

    """
    remote = SshMachine(gpu['server'])
    r_runDocker = remote[remote.cwd / project_dir / "docker/run.sh"]
    r_buildDocker = remote[remote.cwd / project_dir / "docker/build.sh"]
    home_dir = remote.cwd

    killRunningSession(remote, session)
    if update_repo:
        updateRepo(remote, project_dir, repo_ssh_string, branch=branch)

    if rebuild_docker:
        # Build docker
        print("Building container...")
        with remote.cwd(home_dir / project_dir):
            r_buildDocker()
            print("Done.")

    with remote.cwd(home_dir / project_dir):
        # r_runDocker(gpu, "code/main.py -p with ./code/conf/openaiEnv.yaml")
        # print('Executing command: ', command)
        r_runDocker(str(gpu['gpu_nr']), session, command)

    remote.close()


def updateRepo(remote, project_dir, repo_ssh_string, branch=None):
    """
    Helper function to pull newest commits to remote repo.
    """
    r_git = remote['git']
    home_dir = remote.cwd
    # Update repository
    try:
        print("Cloning repository...")
        r_git("clone", repo_ssh_string)
    except plumbum.commands.ProcessExecutionError as e:
        if e.stderr.find('already exists') == -1:
            raise e
        else:
            print("Git directory already exists. Fetching new commits...")
        with remote.cwd(home_dir / project_dir):
            r_git('checkout', 'master')
            r_git('fetch', '-q', 'origin')
            r_git('reset', '--hard', 'origin/master', '-q')

    if branch is not None:

        with remote.cwd(home_dir / project_dir):
            try:
                r_git('checkout', '-b', branch, f'origin/{branch}')
            except plumbum.commands.ProcessExecutionError as e:
                if e.stderr.find('already exists') == -1:
                    raise e
                else:
                    print("Branch already exists")
                    r_git('reset', '--hard', f'origin/{branch}', '-q')

    # Check that we have the same git hash remote than local
    with remote.cwd(home_dir / project_dir):
        r_head = r_git('rev-parse', 'HEAD')
    l_head = git('rev-parse', 'HEAD')
    assert l_head == r_head, "Local git hash != pushed git hash. Did you forget to push changes?"


def killRunningSession(remote, session):
    """
    Help function to kill running containers to start new ones with the same name.
    """
    r_docker = remote['docker']
    try:
        r_docker("stop", session)
        r_docker("rm", session)
        print("Killed running session {}".format(session))
    except plumbum.commands.ProcessExecutionError as e:
        print(e)
        pass


def getSession(project_name, gpu, unique_id):
    """
    Create a name for the container to be started.
    Returns "<project_name><gpu_nr+1>".
    The +1 is to start counting at 1 because 0 creates weird problems.
    """
    nr = gpu['gpu_nr'] + 1
    nr += int(gpu['occupation']) * 10
    nr += unique_id * 100
    # if gpu['double']:
    #     nr += 10
    session = "{}-{}".format(project_name, nr)
    return session


if __name__ == "__main__":
    args = parser.parse_args()
    servers = ['dgx1', 'gandalf', 'gollum', 'sauron', 'brown', 'savitar', 'woma', 'mulga', 'whip']

    if args.clearall:
        clearServers(servers,
                     project_dir="babyai-up",
                     repo_ssh_string='git@github.com:oxwhirl/babyai-up.git')
    else:
        share_with = ['maxigl', 'max', 'grehar', 'tabhid', 'tabz', 'kazami']
        gpus = inspect_gpus(servers=servers, verbose=args.verbose, own_username='grehar',
                            allow_lightly_used_gpus=True, share_with=share_with,
                            max_nr_processes=args.max_nr_processes, upper_gpu_util_threshold=70,
                            upper_memory_threshold=5000, average=args.average)
        print(gpus)
