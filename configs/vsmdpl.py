
from ml_collections import config_dict

def get_config():

    cfg = config_dict.ConfigDict()
    cfg.seed = 42

    cfg.workdir = "/mnt/ceph/users/tnguyen/florah/datasets/processed"
    cfg.name = "VSMDPL-NancMax"
    cfg.num_jobs = 10
    cfg.id_job = 0

    cfg.data = data = config_dict.ConfigDict()
    data.root = "/mnt/ceph/users/tnguyen/florah/datasets/raw_datasets"
    data.name = "VSMDPL-NancMax"
    data.box = "VSMDPL"
    data.num_subbox_dim = 10
    data.num_files_max = int(5**3)

    cfg.preprocess = preprocess = config_dict.ConfigDict()
    preprocess.node_props = ['log_mass', 'log_cvir', 'aexp']
    preprocess.snapshot_step_min = 4
    preprocess.snapshot_step_max = 4
    preprocess.num_subtrees = 1
    preprocess.z_max = 10
    preprocess.num_max_ancestors = 3  # the max cardinality of each node

    return cfg
