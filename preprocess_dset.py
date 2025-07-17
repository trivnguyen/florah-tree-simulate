
import os
import h5py
import sys
import pickle
from pathlib import Path

sys.path.append('/mnt/home/tnguyen/projects/florah/florah-simulate')

import numpy as np
import pandas as pd
from tqdm import tqdm
from absl import flags
from ml_collections import config_flags, config_dict

from florah_analysis import utils, preprocess, sampling
from florah_analysis import tree_utils

def read_dataset(path, features_list=[], to_array=True):
    """ Read dataset from path """
    with h5py.File(path, 'r') as f:
        # read dataset attributes
        headers = dict(f.attrs)
        if len(features_list) == 0:
            features_list = headers['all_features']

        # read pointer to each tree
        ptr = f['ptr'][:]

        # read node features
        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                feat = f[key][:]
                node_features[key] = [
                    feat[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]

        # read tree features
        tree_features = {}
        for key in headers['tree_features']:
            if key in features_list:
                tree_features[key] = f[key][:]

    if to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}

    return node_features, tree_features, headers

def preprocess_dset(config: config_dict.ConfigDict):
    """ Preprocess the raw dataset into a format that can be used for training.
    """

    np.random.seed(config.seed)

    # Read in the raw dataset
    node_features, tree_features = {}, {}

    num_files_read = 0
    for id in range(config.data.num_subbox_dim ** 3):
        i = id // (config.data.num_subbox_dim ** 2)
        j = (id // config.data.num_subbox_dim) % config.data.num_subbox_dim
        k = id % config.data.num_subbox_dim
        dset_path = os.path.join(
            config.data.root, config.data.name, 'isotree_{}_{}_{}.h5'.format(i, j, k))

        if not os.path.exists(dset_path):
            print('Dataset {} not exist. Skipping...'.format(dset_path))
            continue
        else:
            print('Reading dataset from {}'.format(dset_path))
            nodes, trees, headers = read_dataset(dset_path)

        for key in nodes.keys():
            if node_features.get(key) is None:
                node_features[key] = []
            node_features[key].append(nodes[key])

        for key in trees.keys():
            if tree_features.get(key) is None:
                tree_features[key] = []
            tree_features[key].append(trees[key])

        num_files_read += 1
        if num_files_read >= config.data.num_files_max:
            break

    # Concatenate all the features
    for key in node_features.keys():
        node_features[key] = np.concatenate(node_features[key])
    for key in tree_features.keys():
        tree_features[key] = np.concatenate(tree_features[key])

    # calculate extra node properties
    node_features = preprocess.calc_derived_node_properties(node_features)

    # get redshift table and calculate the maximum length of the tree
    snaps, aexp_snaps, z_snaps = utils.read_snapshot_times(config.data.box)

    # Preprocess the dataset
    num_trees = len(tree_features['num_nodes'])
    if z_snaps[-1] > config.preprocess.z_max:
        num_snap_max = np.where(z_snaps > config.preprocess.z_max)[0][0]  # maximum number of snapshots
    else:
        num_snap_max = len(z_snaps) - 1

    # split into multiple jobs
    id_trees_arr = np.array_split(
        np.arange(num_trees), config.num_jobs)[config.id_job]
    loop = tqdm(id_trees_arr, desc='Processing trees', miniters=1000)

    graphs_ppr = []
    node_props = config.preprocess.node_props + ['Snap_num']
    for itree in loop:
        loop.set_description('Processing tree {}'.format(itree))

        node_feats = np.stack([node_features[p][itree] for p in node_props], axis=1)
        halo_ids = node_features['id'][itree]
        halo_desc_ids = node_features['desc_id'][itree]
        snap_nums = node_features['Snap_num'][itree].astype(int)

        for _ in range(config.preprocess.num_subtrees):
            snap_ids = sampling.sample_cumulative_step(
                num_snap_max,
                config.preprocess.snapshot_step_min,
                config.preprocess.snapshot_step_max,
                ini=0)
            snap_ids = snap_nums[0] - snap_ids
            new_halo_ids, new_halo_desc_ids, new_node_feats = tree_utils.subsample_trees(
                halo_ids, halo_desc_ids, node_feats, snap_nums, snap_ids)
            if len(new_halo_ids) < config.preprocess.num_min_nodes:
                continue

            new_halo_ids, new_halo_desc_ids, new_node_feats, new_node_pos = tree_utils.process_progenitors(
                new_halo_ids,
                new_halo_desc_ids,
                10**new_node_feats[..., 0],
                new_node_feats,
                num_max_prog=config.preprocess.num_max_progenitors,
                min_mass_ratio=config.preprocess.min_mass_ratio,
            )
            num_ancestors = tree_utils.calc_num_progenitors(
                new_halo_ids, new_halo_desc_ids)

            # # shuffle the trees
            # if shuffle_nodes:
            #     shuffle_indices = np.random.permutation(len(new_halo_ids))
            #     new_halo_ids = new_halo_ids[shuffle_indices]
            #     new_halo_desc_ids = new_halo_desc_ids[shuffle_indices]
            #     new_node_feats = new_node_feats[shuffle_indices]

            # create the graph
            snap_num = new_node_feats[:, -1]
            new_node_feats = new_node_feats[:, :-1]

            feat_dict = {
                'x': new_node_feats,
                'halo_id': new_halo_ids,
                'halo_desc_id': new_halo_desc_ids,
                'num_ancestors': num_ancestors,
                'prog_pos': new_node_pos,
                'snap': snap_num,
            }
            graph = tree_utils.create_pyg_graph(new_halo_ids, new_halo_desc_ids, feat_dict)

            if not tree_utils.check_mass_sort(graph):
                Warning('Tree {} is not sorted by mass'.format(itree))
                continue

            graphs_ppr.append(graph)

    # save the graphs
    out_path = os.path.join(
        config.workdir, config.name, f'data.{config.id_job}.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print('Saving {} graphs to {}'.format(len(graphs_ppr), out_path))
    with open(out_path, 'wb') as f:
        pickle.dump(graphs_ppr, f)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the preprocessing configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    preprocess_dset(config=FLAGS.config)
