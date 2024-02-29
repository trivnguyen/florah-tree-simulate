
import numpy as np


def squeeze_array(data):
    ptr = np.cumsum([len(d) for d in data])
    ptr = np.insert(ptr, 0, 0)
    new_data = np.concatenate(data)
    return new_data, ptr

def unsqueeze_array(data, ptr):
    new_data = [data[ptr[i]:ptr[i+1]] for i  in range(len(ptr)-1)]
    new_data = np.array(new_data, dtype='object')
    return new_data

def calc_derived_node_properties(node_features):
    """
    Convinience function to calculate derived node properties from existing
    Current properties:
        - log_mass: calculate from mass
        - cvir and log_cvir: calculate from rvir and rs
        - aexp: calculate from redshift
    """
    # calculate log mass
    mass, ptr = squeeze_array(node_features['mass'])
    node_features['log_mass'] = unsqueeze_array(np.log10(mass), ptr)

    # calculate the DM concentration
    rvir, ptr = squeeze_array(node_features['rvir'])
    rs, _ = squeeze_array(node_features['rs'])
    cvir = rvir / rs
    node_features['cvir'] = unsqueeze_array(cvir, ptr)
    node_features['log_cvir'] = unsqueeze_array(np.log10(cvir), ptr)

    # calculate scale factor from redshift
    zred, ptr = squeeze_array(node_features['redshift'])
    zred[zred < 0] = 0   # set negative redshift to 0
    node_features['aexp'] = unsqueeze_array(1 / (1 + zred), ptr)

    return node_features
