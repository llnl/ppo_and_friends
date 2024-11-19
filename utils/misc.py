import numpy as np
import functools
import torch
from ppo_and_friends.utils.stats import RunningMeanStd
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple
import gymnasium.spaces as spaces
import ppo_and_friends.utils.spaces as ppoaf_spaces
import os
import sys
import pickle
from ppo_and_friends.utils.mpi_utils import rank_print
from mpi4py import MPI

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def get_space_dtype_str(action_space):
    """
    Get the data type of a gymnasium space.

    Parameters:
    -----------
    env: environment object
        The action space to query.

    Returns:
    --------
    A string representing the action space dtype.
    """
    if isinstance(action_space, Tuple):
        return "mixed"

    elif np.issubdtype(action_space.dtype, np.floating):
        return "continuous"

    elif np.issubdtype(action_space.dtype, np.integer):
        if type(action_space) == Discrete:
            return "discrete"

        elif type(action_space) == MultiBinary:
            return "multi-binary"

        elif type(action_space) == MultiDiscrete:
            return "multi-discrete"

    return "unknown"


def update_optimizer_lr(optim, lr):
    """
        Update the learning rate of an optimizer.

        Arguments:
            optim    The optimizer to update.
            lr       The new learning rate.
    """
    for group in optim.param_groups:
        group['lr'] = lr


class RunningStatNormalizer(object):
    """
        A structure that allows for normalizing and de-normalizing
        data based on running stats.
    """

    def __init__(self,
                 name,
                 device,
                 test_mode = False,
                 epsilon   = 1e-8):
        """
            Arguments:
                name        The name of the structure (used for saving/loading).
                device      The device where data should be transfered to.
                epsilon     A very small number to help avoid 0 errors.
        """
        self.device        = device
        self.name          = name
        self.test_mode     = test_mode
        self.running_stats = RunningMeanStd()
        self.epsilon       = torch.tensor([epsilon], dtype=torch.float32).to(device)

    def normalize(self,
                  data,
                  update_stats = True,
                  gather_stats = True):
        """
            Normalize incoming data and potential update our stats.

            Arguments:
                data           The data to normalize.
                update_stats   Whether or not to update our runnign stats.
                gather_stats   If update_stats is True, we can need to
                               decide whether or not to gather data across
                               processors before computing our stats.

            Returns:
                The normalized data.
        """
        if update_stats:
            self.running_stats.update(
                data.detach().cpu().numpy(),
                gather_stats)

        mean     = torch.tensor(self.running_stats.mean, dtype=torch.float32).to(self.device)
        variance = torch.tensor(self.running_stats.variance, dtype=torch.float32).to(self.device)

        data = (data - mean) / torch.sqrt(variance + self.epsilon)

        return data

    def denormalize(self,
                    data):
        """
            Denormalize incoming data.

            Arguments:
                data    The data to denormalize.

            Returns:
                The denormalized data.
        """
        mean     = torch.tensor(self.running_stats.mean, dtype=torch.float32)
        variance = torch.tensor(self.running_stats.variance, dtype=torch.float32)
        data     = mean + (data * torch.sqrt(variance + self.epsilon))

        return data

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        if self.test_mode:
            return

        f_name   = "{}_stats_{}.pickle".format(self.name, rank)
        out_file = os.path.join(path, f_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        if self.test_mode:
            f_name  = "{}_stats_0.pickle".format(self.name)
        else:
            f_name  = "{}_stats_{}.pickle".format(self.name, rank)

        in_file = os.path.join(path, f_name)

        #
        # There are cases where we initially train using X ranks, and we
        # later want to continue training using (X+k) ranks. In these cases,
        # let's copy rank 0's info to all ranks > X.
        #
        if not os.path.exists(in_file):
            f_name  = "{}_stats_0.pickle".format(self.name)
            in_file = os.path.join(path, f_name)

        with open(in_file, "rb") as fh:
            self.running_stats = pickle.load(fh)

def format_seconds(seconds):
    """
        Format a floating point representing seconds into
        a nice readable string.

        Arguments:
            seconds    The seconds to format.

        Returns:
            A formatted string as either seconds, minutes, or hours.
    """

    output_time = seconds
    output_unit = "seconds"

    if output_time >= 60.:
        output_time /= 60.
        output_unit = "minutes"

        if output_time >= 60.:
            output_time /= 60.
            output_unit = "hours"

    return "{:.2f} {}".format(output_time, output_unit)


def get_space_shape(space):
    """
    Return a hand-wavy shape of a given gymnasium space. Not
    all spaces have a "shape" attribute, but we can infer what
    it realistically would be.

    Parameters
    ----------
    space: gymnasium space
        The space to get the shape of.

    Returns
    -------
    int
        An inferred shape of the space.
    """
    if isinstance(space, Box):
        return space.shape

    elif isinstance(space, Discrete):
        return (1,)

    elif isinstance(space, MultiBinary):
        return (space.n,)

    elif (isinstance(space, MultiDiscrete) or
        isinstance(space, ppoaf_spaces.FlatteningDict) or
        isinstance(space, ppoaf_spaces.FlatteningTuple) or
        isinstance(space, ppoaf_spaces.SparseFlatteningDict) or
        isinstance(space, ppoaf_spaces.SparseFlatteningTuple)):
        return space.shape

    elif isinstance(space, Tuple):

        space_shapes = []
        for sub_space in space:
            space_shapes.append(get_space_shape(sub_space))

        for i in range(1, len(space_shapes)):
            if len(space_shapes[i]) > 1:
                msg  = "ERROR: the sub-spaces of mixed action spaces must "
                msg += "have a length of 1 but received shapes: "
                msg += f"{space_shapes}"
                rank_print(msg)
                comm.Abort()

        return tuple(np.array(space_shapes).sum(axis=0))

    else:
        msg  = f"ERROR: unsupported space, {type(space)}, encountered in "
        msg += "get_space_shape."
        rank_print(msg) 
        comm.Abort()


def get_flattened_space_length(space):
    """
    Get the length of a flattened gymnasium space. Only some spaces
    are supported here.

    Parameters
    ----------
    space: gymnasium space
        The space to get the flattened length of.

    Returns:
    int
        The length of the gymnasium space.
    """
    space_shape = get_space_shape(space)
    return functools.reduce(lambda a, b: a*b, space_shape)


def get_size_and_shape(descriptor):
    """
    Given a shape/size descriptor as either a tuple or int,
    return the associated shape and size.

    Parameters:
    -----------
    descriptor: tuple or int
        An int or tuple representing the size/shape.

    Returns:
    --------
    tuple:
        The size and shape as (int, tuple).
    """
    desc_type = type(descriptor)
    assert desc_type == tuple or desc_type == int or desc_type == np.ndarray

    if desc_type == tuple or desc_type == np.ndarray:
        size  = functools.reduce(lambda a, b: a*b, descriptor)
        shape = descriptor
    else:
        size  = descriptor
        shape = (descriptor,)

    return size, shape


def get_action_prediction_shape(space):
    """
    Assuming the space is an action space that our actor
    is predicting, get the shape of this prediction.

    Parameters:
    -----------
    space: gymnasium space
        The action space to infer the shape from.

    Returns:
    --------
    tuple:
        The shape of our actor's prediction.
    """
    if isinstance(space, Box):
        return space.shape

    elif (isinstance(space, Discrete) or
        isinstance(space, MultiBinary)):
        return (space.n,)

    elif isinstance(space, MultiDiscrete):
        return (functools.reduce(lambda a, b: a+b, space.nvec),)

    elif isinstance(space, Tuple):

        pred_shapes = []
        for sub_space in space:
            pred_shapes.append(get_action_prediction_shape(sub_space))

        for i in range(1, len(pred_shapes)):
            if len(pred_shapes[i]) > 1:
                msg  = "ERROR: the sub-spaces of mixed action spaces must "
                msg += "have a length of 1 but received prediction shapes: "
                msg += f"{pred_shapes}"
                rank_print(msg)
                comm.Abort()

        return tuple(np.array(pred_shapes).sum(axis=0))

    else:
        msg  = f"ERROR: unsupported space, {type(space)}, encountered in "
        msg += "get_action_prediciont_shape."
        rank_print(msg) 
        comm.Abort()


def get_agent_shared_space(space, num_agents):
    """
    Get a version of the given space that spans all agents.
    NOTE: it is assumed that all agents share the same space.

    Parameters:
    -----------
    space: gymnasium space
        The space to expand for all agents.
    num_agents: int
        The number of agents to expand the space for.

    Returns:
    --------
    gymnasium space:
        The given space expanded to include all agents.
    """
    if type(space) == Box:
        box_spaces = Tuple([space for _ in range(num_agents)])
        return spaces.utils.flatten_space(box_spaces)

    elif type(space) == Discrete:
        if not np.issubdtype(type(space.n), np.integer):
            msg  = f"ERROR: expected space.n to be of type int for Discrete "
            msg += f"but received {space.n} of type {type(space.n)}."
            rank_print(msg)
            comm.Abort()

        return MultiDiscrete([space.n] * num_agents, dtype=space.dtype)

    elif type(space) == MultiDiscrete:
        return MultiDiscrete(np.tile(space.nvec, num_agents),
            start=np.tile(space.start, num_agents))

    elif type(space) == MultiBinary:
        if not np.issubdtype(type(space.n), np.integer):
            msg  = f"ERROR: expected space.n to be of type int for MultiBinary "
            msg += f"but received {space.n} of type {type(space.n)}."
            rank_print(msg)
            comm.Abort()

        return MultiBinary(space.n * num_agents)

    else:
        msg  = f"ERROR: unsupported space of type {type(space)} sent "
        msg += "to policy.get_agent_shared_space."
        rank_print(msg)
        comm.Abort()
