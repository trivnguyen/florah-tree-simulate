
import argparse
import os
import time
from pathlib import Path

import florah
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import ytree
from florah_analysis import utils

ALL_NODE_PROPS = [
    'mass', 'redshift', 'rvir', 'vrms', 'vmax', 'rs', 'x', 'y', 'z',
    'id', 'desc_id', 'Snap_num']
DEFAULT_METADATA_DIR = "/mnt/ceph/users/tnguyen/florah-tree/metadata"
DEFAULT_ISOTREE_DIR = "/mnt/home/tnguyen/isotrees"
DEFAULT_RAW_DATASET_DIR = "/mnt/home/tnguyen/ceph/florah-tree/datasets/raw_datasets"

def get_ancestors(halo, node_props, branch_id=0, min_mass=0, num_ancestors_max=1):
    """ Get full halo trees """

    features = {p: [np.array(halo['prog', p]), ] for p in node_props}
    branch_indices = [branch_id, ]

    for prog in list(halo['prog']):
        ancestors = list(prog.ancestors)
        mass = np.array([anc['mass'] for anc in ancestors])
        sorted = np.argsort(mass)[::-1]

        for i in sorted[1:num_ancestors_max+1]:
            anc = ancestors[i]
            if anc['mass'] >= min_mass:
                next_branch_id = branch_id + i + 1
                next_features, next_branch_indices = get_ancestors(
                    anc, node_props, next_branch_id, min_mass, num_ancestors_max)
                for p in node_props:
                    features[p] += next_features[p]
                branch_indices += next_branch_indices

    return features, branch_indices


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--box-name", required=True, type=str,
        help="Input name of the box to create the dataset for")
    parser.add_argument(
        "--box-name-out", required=True, type=str,
        help="Output name of the dataset")
    parser.add_argument(
        "--num-anc-max", default=1, required=False, type=int,
        help="Maximum number of ancestors to include in the dataset")
    parser.add_argument(
        "--min-num-root", default=500, required=False, type=int,
        help="Minimum number of DM particles in a root halo")
    parser.add_argument(
        "--min-num-halo", default=100, required=False, type=int,
        help="Minimum number of DM particles in a prog halo")
    parser.add_argument(
        "--ijob", required=False, type=int, default=0,
        help="Job number")
    return parser.parse_args()


def main():
    FLAGS = parse_cmd()

    # Read in isotree
    isotree_dir = os.path.join(envs.DEFAULT_ISOTREE_DIR, FLAGS.box_name)
    outdir = os.path.join(envs.DEFAULT_RAW_DATASET_DIR, FLAGS.box_name_out)
    os.makedirs(outdir, exist_ok=True)

    # Read in metadata
    meta = pd.read_csv(
        os.path.join(envs.DEFAULT_METADATA_DIR, 'meta.csv'),
        sep=',', header=0)
    Mdm = float(meta['Mdm'][meta['name']==FLAGS.box_name])
    Mmin_root = Mdm * FLAGS.min_num_root
    Mmin_halo = Mdm * FLAGS.min_num_halo
    num_subbox_dim = int(meta['num_subbox_dim'][meta['name']==FLAGS.box_name])

    print("Mass resolution: {}".format(Mdm))
    print("Minimum root mass: {}".format(Mmin_root))
    print("Minimum root number: {}".format(FLAGS.min_num_root))
    print("Minimum halo mass: {}".format(Mmin_halo))
    print("Minimum halo number: {}".format(FLAGS.min_num_halo))
    print("box_name: {}".format(FLAGS.box_name))
    print("box_name_out: {}".format(FLAGS.box_name_out))

    # Get the file name
    # NOTE: this is a bit hacky (and lazy) but it works
    tree_files = []
    for i in range(num_subbox_dim):
        for j in range(num_subbox_dim):
            for k in range(num_subbox_dim):
                path = os.path.join(
                    isotree_dir, f"isotree_{i}_{j}_{k}.dat")
                if not os.path.exists(path):
                    print(f"{path} not exist.  skipping...")
                    continue
                else:
                    print(path)
                tree_files.append(path)
    tree_fn = tree_files[FLAGS.ijob]
    iso_tree_name = Path(tree_fn).stem

    # Read in the data file
    print('Reading tree from {}'.format(tree_fn))

    data = ytree.load(tree_fn)
    if ('VSMDPL' in FLAGS.box_name):
        data.add_alias_field("rvir", "Rvir")
    elif ('TNG' in FLAGS.box_name):
        data.add_alias_field("rvir", "Rvir")
        data.add_alias_field('Snap_num', 'Snap_idx')

    elif ('GUREFT' in FLAGS.box_name):
        data.add_alias_field("mass", "mvir")

    # apply a minimum mass cut on the root halos
    mass = data['mass'].value
    indices = np.arange(len(mass))
    # indices = np.argsort(mass)[::-1]  # sort by mass, good for testing
    indices = indices[mass[indices] >= Mmin_root]
    tree_list = list(data[indices])

    node_features = {p: [] for p in ALL_NODE_PROPS}
    tree_features = {'root_id': [], 'num_nodes': []}
    for itree in range(len(tree_list)):
        if itree % 100 == 0:
            print('Processing tree {} / {}'.format(itree, len(tree_list)))

        halo = tree_list[itree]
        features, branch_indices = get_ancestors(
            halo, ALL_NODE_PROPS, branch_id=0, min_mass=Mmin_halo,
            num_ancestors_max=FLAGS.num_anc_max)
        features = {p: np.concatenate(features[p]) for p in ALL_NODE_PROPS}
        num_nodes = len(features['mass'])
        for p in ALL_NODE_PROPS:
            node_features[p].append(features[p])

        # set the tree_id to the halo ID of the root halo
        tree_features['root_id'].append(halo['id'])
        tree_features['num_nodes'].append(len(features['mass']))

    # concatenate all the arrays
    for p in ALL_NODE_PROPS:
        node_features[p] = np.array(node_features[p], dtype="object")
    for p in tree_features.keys():
        tree_features[p] = np.array(tree_features[p])

    # create pointer array
    ptr = np.cumsum(tree_features['num_nodes'])
    ptr = np.insert(ptr, 0, 0)

    # create headers
    headers = {
        'node_features': ALL_NODE_PROPS,
        'tree_features': list(tree_features.keys()),
        'num_node_features': len(ALL_NODE_PROPS),
        'num_tree_features': len(tree_features.keys()),
        'num_trees': len(tree_features['root_id']),
        'num_nodes': np.sum(tree_features['num_nodes']),
        'min_root_mass': Mmin_root,
        'min_halo_mass': Mmin_halo,
        'num_ancestors_max': FLAGS.num_anc_max,
    }

    out_tree_fn = os.path.join(outdir, f"{iso_tree_name}.h5")
    print("Writing to {}".format(out_tree_fn))
    florah.utils.io.write_dataset(
        out_tree_fn, node_features,
        tree_features, ptr=ptr, headers=headers)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Time taken: {} s".format(t2-t1))
    print("Done!")