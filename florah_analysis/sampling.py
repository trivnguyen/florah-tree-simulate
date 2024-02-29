
import numpy as np


def sample_cumulative_step(num_step_max, step_size_min, step_size_max, ini=0):
    """ Sample cumulative steps. The number of steps is randomly sampled
    between step_size_min and step_size_max. The first step is always ini.

    Parameters
    ----------
    num_step_max : int
        Maximum number of steps
    step_size_min : int
        Minimum step size
    step_size_max : int
        Maximum step size
    step_ini : int
        The starting step
    """
    if ini > num_step_max:
        raise ValueError('ini must be smaller than num_step_max')

    # randomly sample the time steps
    step_sizes = np.random.randint(
        step_size_min, step_size_max + 1, size=num_step_max//step_size_min)
    step_sizes = np.insert(step_sizes, 0, ini)

    # get the indices
    steps = np.cumsum(step_sizes)
    steps = steps[steps < num_step_max]
    return steps
