
from ml_collections import config_dict

def get_config():

    cfg = config_dict.ConfigDict()
    cfg.seed = 50

    cfg.workdir = "/mnt/ceph/users/tnguyen/florah-tree/datasets/processed"
    cfg.name = "gureft90-nprog3-dt1_4"
    cfg.num_jobs = 1
    cfg.id_job = 0

    cfg.data = data = config_dict.ConfigDict()
    data.root = "/mnt/ceph/users/tnguyen/florah-tree/datasets/raw_datasets"
    data.name = "gureft90-nprogMAX"
    data.name = "gureft90-nprogMAX"
    data.box = "gureft90"
    data.num_subbox_dim = 1
    data.num_files_max = int(1**3)

    cfg.preprocess = preprocess = config_dict.ConfigDict()
    preprocess.node_props = ['log_mass', 'log_cvir', 'aexp']
    preprocess.snapshot_step_min = 1
    preprocess.snapshot_step_max = 4
    preprocess.num_subtrees = 5
    preprocess.min_mass_ratio = 0.001
    preprocess.z_max = 8
    preprocess.num_max_progenitors = 3  # the max cardinality of each node

    return cfg
