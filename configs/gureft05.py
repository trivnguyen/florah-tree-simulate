
from ml_collections import config_dict

def get_config():

    cfg = config_dict.ConfigDict()
    cfg.seed = 42

    cfg.workdir = "/mnt/ceph/users/tnguyen/florah-tree/datasets/processed"
    cfg.name = "gureft05-nanc2-massiveProgs"
    cfg.num_jobs = 10
    cfg.id_job = 0

    cfg.data = data = config_dict.ConfigDict()
    data.root = "/mnt/ceph/users/tnguyen/florah-tree/datasets/raw_datasets"
    data.name = "GUREFT05-NancMax"
    data.box = "GUREFT05"
    data.num_subbox_dim = 2
    data.num_files_max = int(2**3)

    cfg.preprocess = preprocess = config_dict.ConfigDict()
    preprocess.node_props = ['log_mass', 'log_cvir', 'aexp']
    preprocess.snapshot_step_min = 4
    preprocess.snapshot_step_max = 4
    preprocess.num_subtrees = 1
    preprocess.min_mass_ratio = 0.01
    preprocess.z_max = 10
    preprocess.num_max_ancestors = 3  # the max cardinality of each node

    return cfg
