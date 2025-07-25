

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from networkx.drawing.nx_pydot import graphviz_layout
from torch_geometric.data import Data

def convert_to_nx(data):
    # convert PyG graph to a networkx graph manually
    # since the to_networkx method is bugged
    G = nx.Graph()
    G.add_nodes_from(range(len(data.x)))
    G.add_edges_from(data.edge_index.T.numpy())
    for i in range(len(data.x)):
        G.nodes[i]['x'] = data.x[i].numpy()
    return G

def create_nx_graph(halo_id, halo_desc_id, halo_props=None):
    """ Create a directed graph of the halo merger tree.

    Parameters
    ----------
    halo_id : array_like
        Array of halo IDs.
    halo_desc_id : array_like
        Array of halo descendant IDs.
    halo_props : dict or None, optional
        Array of halo properties. If provided, the properties will be added as
        node attributes.

    Returns
    -------
    G : networkx.DiGraph
        A directed graph of the halo merger tree.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes using indices
    for idx in range(len(halo_id)):
        if halo_props is not None:
            prop_dict = {key: halo_props[key][idx] for key in halo_props.keys()}
            G.add_node(idx, **prop_dict)
        G.add_node(idx)

    # Add edges based on desc_id
    for idx, desc_id in enumerate(halo_desc_id):
        if desc_id != -1:
            parent_idx = np.where(halo_id==desc_id)[0][0] # Find the index of the parent ID
            G.add_edge(parent_idx, idx) # Use indices for edges
    return G


def create_pyg_graph(halo_id, halo_desc_id, halo_props=None):
    """ Create a directed graph of the halo merger tree using PyTorch Geometric.

    Parameters
    ----------
    halo_id : array_like
        Array of halo IDs.
    halo_desc_id : array_like
        Array of halo descendant IDs.
    halo_props : dict or None, optional
        Dictionary of halo properties. If provided, the properties will be added as
        node features.

    Returns
    -------
    data : torch_geometric.data.Data
        A PyTorch Geometric Data object representing the halo merger tree.
    """
    # Convert halo_id and halo_desc_id to torch tensors
    halo_id_tensor = torch.tensor(halo_id, dtype=torch.long)
    halo_desc_id_tensor = torch.tensor(halo_desc_id, dtype=torch.long)

    # Create edge index
    edge_index = []
    for idx, desc_id in enumerate(halo_desc_id):
        if desc_id != -1:
            parent_idx = (halo_id_tensor == desc_id).nonzero(as_tuple=True)[0][0]
            edge_index.append([parent_idx, idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Handle node features (halo properties)
    if halo_props is not None:
        # convert each property to a tensor
        props = {}
        for key in halo_props.keys():
            # NOTE: ancestors here are the same as progenitors
            if key in ("halo_id", "halo_desc_id", "num_ancestors"):
                dtype = torch.long
            else:
                dtype = torch.float
            props[key] = torch.tensor(halo_props[key], dtype=dtype)

        # Assuming halo_props is a dict of lists
        # x = [torch.tensor(halo_props[key], dtype=torch.float) for key in halo_props.keys()]
        # x = torch.stack(x, dim=1)  # Convert to a 2D tensor
    else:
        props = {}

    # Create PyG Data object
    data = Data(edge_index=edge_index, **props)

    return data

def plot_graph(G, **kwargs):
    pos = graphviz_layout(G, prog='dot')
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_args = dict(
        with_labels=False,
        node_size=20, node_color="black",
        font_size=10, font_color="black")
    draw_args.update(kwargs)
    nx.draw(G, pos, ax=ax, **draw_args)
    ax.set_title("Merger Tree Graph")
    plt.show()
    return fig, ax

def get_main_branch(mass, edge_index):
    # follow the main branch and get index
    main_branch_index = [0]
    curr_index = 0
    while True:
        # get all progenitors
        prog_index = edge_index[1][edge_index[0] == curr_index]

        # if no progenitors, break
        if len(prog_index) == 0:
            break

        # get the progenitor with the highest mass
        prog_mass = mass[prog_index]
        prog_index = prog_index[prog_mass.argmax()].item()

        main_branch_index.append(prog_index)
        curr_index = prog_index

    return main_branch_index

def check_mass_sort(tree):
    """ Check if the mass is sorted correctly within the tree """
    halo_idx, num_progs = torch.unique(tree.edge_index[0], return_counts=True)

    is_sorted = True
    for idx in halo_idx:
        progs_idx = tree.edge_index[1][tree.edge_index[0] == idx]
        progs_x = tree.x[progs_idx]

        # Check if progs_x[:, 0] is sorted in descending order
        is_sorted_prog = torch.all(progs_x[:, 0][:-1] >= progs_x[:, 0][1:])
        is_sorted = is_sorted & is_sorted_prog
        if not is_sorted_prog:
            print(f"Halo {idx} is not sorted correctly")
            print(tree.x[idx], progs_x, progs_idx)
    return is_sorted

def subsample_trees(
    halo_ids, halo_desc_ids, node_feats, snap_nums, new_snap_nums):
    """ Subsample the trees to only include the snapshots in new_snap_nums.
    """
    new_node_feats = []
    new_halo_ids = []
    new_halo_desc_ids = []
    for i in range(len(snap_nums)):
        if snap_nums[i] in new_snap_nums:
            new_node_feats.append(node_feats[i])
            new_halo_ids.append(halo_ids[i])
            new_halo_desc_ids.append(halo_desc_ids[i])
    if len(new_node_feats) == 0:
        return new_node_feats, new_halo_ids, new_halo_desc_ids
    new_node_feats = np.stack(new_node_feats, axis=0)
    new_halo_ids = np.array(new_halo_ids)
    new_halo_desc_ids = np.array(new_halo_desc_ids)

    # update the halo_desc_ids
    # iterate over all halos
    id_to_index = {halo_id: i for i, halo_id in enumerate(halo_ids)}
    for i in range(len(new_halo_desc_ids)):
        halo_desc_id = new_halo_desc_ids[i]
        if halo_desc_id == -1:
            continue
        while halo_desc_id not in new_halo_ids:
            halo_desc_id = halo_desc_ids[id_to_index[halo_desc_id]]
        new_halo_desc_ids[i] = halo_desc_id

    return new_halo_ids, new_halo_desc_ids, new_node_feats

def calc_num_progenitors(halo_ids, halo_desc_ids):
    """ Calculate the number of progenitors for each halo. """
    unique_desc_ids, counts = np.unique(halo_desc_ids, return_counts=True)
    num_progenitors = np.zeros(len(halo_desc_ids), dtype=np.int32)
    for desc_id, count in zip(unique_desc_ids, counts):
        num_progenitors[halo_ids == desc_id] = count
    return num_progenitors

def process_progenitors(
    halo_ids, halo_desc_ids, halo_mass, node_feats,
    num_max_prog=1, min_mass_ratio=0.01, return_prog_position=True
    ):
    """ Process the progenitors of each halo by sorting them by mass and removin
    any progenitors that are not the most massive or have a mass ratio less than
    min_mass_ratio. """

    def fun(index, halo_ids, halo_desc_ids, halo_mass, num_max_prog=1, min_mass_ratio=0.01, prog_pos=0):
        """ Recursively sort the progenitors of a halo by mass and return the sorted indices. """

        # Get the current halo ID and indices of its progenitors
        halo_id = halo_ids[index]
        halo_desc_indices = np.where(halo_desc_ids == halo_id)[0]

        # Start with the current halo index in the sorted list
        sorted_index = [index, ]
        prog_position = [prog_pos, ]

        if len(halo_desc_indices) == 0:
            return sorted_index, prog_position

        # Sort the progenitors by mass in descending order
        sort = np.argsort(halo_mass[halo_desc_indices])[::-1]
        sorted_desc_indices = halo_desc_indices[sort]
        halo_desc_mass = halo_mass[sorted_desc_indices]
        halo_desc_mass_ratio = halo_desc_mass / np.max(halo_desc_mass)

        mask = halo_desc_mass_ratio > min_mass_ratio
        sorted_desc_indices = sorted_desc_indices[mask][:num_max_prog]

        # Recursively process each progenitor and gather their sorted indices
        for i, progenitor_index in enumerate(sorted_desc_indices):
            s, p = fun(
                progenitor_index,
                halo_ids,
                halo_desc_ids,
                halo_mass,
                num_max_prog=num_max_prog,
                min_mass_ratio=min_mass_ratio,
                prog_pos=i
            )
            sorted_index.extend(s)
            prog_position.extend(p)

        return sorted_index, prog_position

    sorted_index, prog_position = fun(
        0, halo_ids, halo_desc_ids, halo_mass,
        num_max_prog=num_max_prog, min_mass_ratio=min_mass_ratio, prog_pos=0)
    new_node_feats = node_feats[sorted_index]
    new_halo_ids = halo_ids[sorted_index]
    new_halo_desc_ids = halo_desc_ids[sorted_index]
    prog_position = np.array(prog_position)

    if return_prog_position:
        return new_halo_ids, new_halo_desc_ids, new_node_feats, prog_position
    else:
        return new_halo_ids, new_halo_desc_ids, new_node_feats


# def remove_progenitors(
#     halo_ids, halo_desc_ids, halo_mass, node_feats, num_max_prog=1,
#     min_mass_ratio=0.01):
#     """ Enforce the maxmium number of progenitors. If more than num_max_prog
#     halos have the same descendant, only keep the num_max_prog most massive ones.

#     If the mass ratio between the progenitors and the most massive progenitor is
#     less than min_mass_ratio, remove the progenitor.

#     Note that this code assumes the halos are sorted by snapshot number starting
#     from the root halo.
#     """
#     if num_max_prog < 1:
#         raise ValueError("num_max_prog must be >= 1.")

#     # identify the halos with the same descendant
#     unique_desc_ids, counts = np.unique(halo_desc_ids, return_counts=True)
#     bad_indices = []
#     for desc_id, count in zip(unique_desc_ids, counts):
#         if count == 1:
#             continue
#         # find the indices of the halos with this desc_id
#         indices = np.where(halo_desc_ids == desc_id)[0]

#         # sort the indices by mass
#         sort = np.argsort(halo_mass[indices])[::-1]
#         indices = indices[sort]
#         m_prog = halo_mass[indices]
#         m_prog_ratio = m_prog / np.max(m_prog)

#         # remove the progenitors with mass ratio < min_mass_ratio
#         bad_mask = np.zeros(len(indices), dtype=bool)
#         bad_mask = bad_mask | (m_prog_ratio < min_mass_ratio)
#         bad_mask[num_max_prog:] = True

#         bad_indices.append(indices[bad_mask])

#     if len(bad_indices) == 0:
#         return halo_ids, halo_desc_ids, node_feats
#     else:
#         bad_indices = np.concatenate(bad_indices)

#     # iterate over all halos and add only good halos
#     # good halos are those that are not in bad_indices and have a valid desc_id
#     new_node_feats = []
#     new_halo_ids = []
#     new_halo_desc_ids = []
#     for i in range(len(halo_ids)):
#         accept = (i not in bad_indices) and np.isin(halo_desc_ids[i], new_halo_ids)
#         accept = accept or (halo_desc_ids[i] == -1)  # always keep the root halo (desc_id = -1)
#         if accept:
#             new_node_feats.append(node_feats[i])
#             new_halo_ids.append(halo_ids[i])
#             new_halo_desc_ids.append(halo_desc_ids[i])
#     new_node_feats = np.stack(new_node_feats, axis=0)
#     new_halo_ids = np.array(new_halo_ids)
#     new_halo_desc_ids = np.array(new_halo_desc_ids)

#     return new_halo_ids, new_halo_desc_ids, new_node_feats