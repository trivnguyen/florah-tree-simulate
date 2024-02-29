
import os
import numpy as np

from . import envs

def read_snapshot_times(box_name):
    """ Read in snapshot times from the simulations """
    default_dir = envs.DEFAULT_METADATA_DIR
    if "GUREFT" in box_name:
        table_name = "snapshot_times_gureft.txt"
    else:
        table_name = "snapshot_times_{}.txt".format(box_name.lower())

    snapshot_times = np.genfromtxt(
        os.path.join(default_dir, table_name), delimiter=',', unpack=True)
    return snapshot_times

def get_run(name, version="best", prefix=envs.DEFAULT_OUTPUT_DIR):
    """ Get path to run directory """
    # check if prefix directory exists
    if not os.path.exists(prefix):
        raise FileNotFoundError(f"prefix directory {prefix} not found")
    # check if run directory exists
    path = os.path.join(prefix, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"run {name} not found in directory {prefix}")
    # check if version exists
    if version != "best":
        version_path = os.path.join(path, version)
        if not os.path.exists(version_path):
            raise FileNotFoundError(
                f"version {version} not found in run {name} with prefix {prefix}")
        return version_path
    else:
        return get_best_run(path)[0]

def get_best_run(run_path):
    """ Get the version path with the best checkpoint """
    # iterate over all version
    min_loss = 100000
    best_version_path = None
    for version in range(1000):
        run_version_path = os.path.join(run_path, f"version_{version}")
        if not os.path.exists(run_version_path):
            break
        best_checkpoint, best_version_loss = get_best_checkpoint(run_version_path)
        if best_version_loss < min_loss:
            min_loss = best_version_loss
            best_version_path = run_version_path
    return best_version_path, min_loss

def get_best_checkpoint(run_version_path):
    checkpoints = glob.glob(
        os.path.join(run_version_path, "checkpoints/epoch*.ckpt"))
    # get the loss of each checkpoint and return min loss and version
    min_loss = 100000
    best_checkpoint = None
    for ckpt in checkpoints:
        temp = Path(ckpt).stem
        loss = float(temp.split('=')[-1])
        if loss < min_loss:
            min_loss = loss
            best_checkpoint = ckpt
    return best_checkpoint, min_loss
