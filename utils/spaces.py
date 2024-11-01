import gym as old_gym
import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict
from gymnasium.spaces.space import Space
from ppo_and_friends.utils.mpi_utils import rank_print
import numpy as np
from typing import Any

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


def validate_observation_space(env):
    """
    """
    is_discrete_space = lambda s : type(s) == Discrete or type(s) == old_gym.spaces.Discrete
    get_space_args    = lambda s : (s.n, s.start, s._np_random)

    if is_discrete_space(env.observation_space):
        n, start, seed = get_space_args(env.observation_space)
        env.observation_space = ShapelyDiscrete(n = n, start = start, seed = seed)

    elif type(env.observation_space) == Tuple:
        new_space = []
        for i in range(len(env.observation_space)):
            if is_discrete_space(env.observation_space[i]):
                n, start, seed = get_space_args(env.observation_space[i])
                new_space.append(ShapelyDiscrete(n = n, start = start, seed = seed))
            else:
                new_space.append(env.observation_space[i])

        env.observation_space = Tuple(new_space)

    elif type(env.observation_space) == Dict:
        for key in env.observation_space:
            if is_discrete_space(env.observation_space[key]):
                n, start, seed = get_space_args(env.observation_space[key])
                env.observation_space[key] = ShapelyDiscrete(n = n, start = start, seed = seed)

    if hasattr(env, "env"):
        env.env = validate_observation_space(env.env)

    return env


def gym_space_to_gymnasium_space(space):
    """
    gym and gymnasium spaces are incompatible. This function
    just converts gym spaces to gymnasium spaces to bypass
    the errors that crop up.

    Parameters:
    -----------
    space: gym space
        The gym space to convert.

    Returns:
    --------
    The input space converted to gymnasium.
    """
    if issubclass(type(space), old_gym.spaces.Box):
        space = gym.spaces.Box(
            low   = space.low,
            high  = space.high,
            shape = space.shape,
            dtype = space.dtype)

    elif issubclass(type(space), old_gym.spaces.Discrete):
        try:
            space = gym.spaces.Discrete(
                n     = space.n,
                start = space.start)
        except:
            space = gym.spaces.Discrete(
                n = space.n)

    elif issubclass(type(space), old_gym.spaces.MultiBinary):
        space = gym.spaces.MultiBinary(
            n = space.n)

    elif issubclass(type(space), old_gym.spaces.MultiDiscrete):
        space = gym.spaces.MultiDiscrete(
            nvec  = space.nvec,
            dtype = space.dtype)

    elif issubclass(type(space), old_gym.spaces.Dict):
        new_space = gym.spaces.Dict()

        for key in space:
            new_space[key] = gym_space_to_gymnasium_space(space[key])

        space = new_space

    elif issubclass(type(space), old_gym.spaces.Tuple):
        new_space = []

        for subspace in space:
            new_space.append(gym_space_to_gymnasium_space(subspace))

        space = gym.spaces.Tuple(new_space)

    elif ((hasattr(old_gym.spaces, "Text") and issubclass(type(space), old_gym.spaces.Text)) or
        (hasattr(old_gym.spaces, "Sequence") and issubclass(type(space), old_gym.spaces.Sequence)) or
        (hasattr(old_gym.spaces, "Graph") and issubclass(type(space), old_gym.spaces.Graph))):
        msg  = f"ERROR: conversion of gym space {space} to "
        msg += f"gymnasium is not currently supported. Contact "
        msg += f"a developer to extend support for this space."
        rank_print(msg)
        comm.abort()

    return space


class FlatteningTuple(Tuple):
    """
    A wrapper around a gymnasium Tuple space that allows us
    to get combined/flattened samples.
    """

    def __init__(self, sub_spaces, *args, **kw_args):
        """
        Parameters:
        -----------
        sub_spaces: iterable
            An iterable containing the sub-spaces to encapsulate.
        """
        self.sample_sizes   = []
        accepted_sub_spaces = [Box, Discrete, MultiDiscrete, MultiBinary]

        old_gym_spaces = [\
            old_gym.spaces.Box,
            old_gym.spaces.Discrete,
            old_gym.spaces.MultiDiscrete,
            old_gym.spaces.MultiBinary]

        for i in range(len(sub_spaces)):
            space = sub_spaces[i]

            if type(space) in old_gym_spaces:
                space = gym_space_to_gymnasium_space(space)
                sub_spaces[i] = space

            if type(space) not in accepted_sub_spaces:
                msg  = f"ERROR: sub space {space} is not currently supported by "
                msg += f"the FlatteningTuple. Supported sub-spaces are "
                msg += f"{accepted_sub_spaces}."
                rank_print(msg)
                comm.Abort()

            #
            # TODO: we could probably support multi-dimensional sub-spaces when
            # space.is_np_flattenable evaluates to True.
            #
            if len(space.shape) > 1:
                msg  = "ERROR: FlatteningTuple does not currently support "
                msg += "sub-spaces with shapes greater than 1. Given space: "
                msg += "{space}."
                rank_print(msg)
                comm.Abort()

            sample = space.sample()

            if type(sample) == np.ndarray:
                self.sample_sizes.append(sample.size)
            else:
                self.sample_sizes.append(1)

        self.sample_sizes   = np.array(self.sample_sizes, dtype=np.int32)
        self.flattened_size = self.sample_sizes.sum()

        super().__init__(sub_spaces, *args, **kw_args)

    def sample(self):
        """
        Sample the space.
        """
        return self.flatten_sample(super().sample())

    def flatten_sample(self, sample):
        """
        Wrap a our sampled

        Parameters:
        -----------
        sample: np.ndarray
            A sample from our tuple space.

        Returns:
        --------
        A flattened version of the sample as an np.ndarray.
        """
        flattened_sample = np.zeros(self.flattened_size)

        start = 0
        for idx, sub_sample in enumerate(sample):
            stop = start + self.sample_sizes[idx]

            flattened_sample[start : stop] = sub_sample

            start = stop

        return flattened_sample

    @property
    def shape(self):
        return (self.flattened_size,)


class ShapelyDiscrete(Discrete):

    def __init__(
        self,
        n,
        seed  = None,
        start = 0,
    ):
        """
        """
        assert np.issubdtype(
            type(n), np.integer
        ), f"Expects `n` to be an integer, actual dtype: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert np.issubdtype(
            type(start), np.integer
        ), f"Expects `start` to be an integer, actual type: {type(start)}"

        self.n     = np.int64(n)
        self.start = np.int64(start)

        super(Discrete, self).__init__((1,), np.int64, seed)

    def sample(self, *args, **kw_args):
        """
        """
        return np.array((super.sample(*args, **kw_args),))


