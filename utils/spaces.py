import gym as old_gym
import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict, Sequence, Graph
from gymnasium.spaces.space import Space
from ppo_and_friends.utils.mpi_utils import rank_print
import numpy as np
from typing import Any
import numbers

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

    else:
        msg  = f"ERROR: conversion of gym space {space} to "
        msg += f"gymnasium is not currently supported. Contact "
        msg += f"a developer to extend support for this space."
        rank_print(msg)
        comm.abort()

    return space

class FlatteningCompositeSpace():

    def __init__(self, *args, auto_flatten=False, **kw_args):
        """
        """
        self.supported_spaces = [
            Discrete,
            MultiDiscrete,
            MultiBinary,
            Box,
            Dict,
            Tuple,
        ]
        self.auto_flatten = auto_flatten

    def flatten_sample(self, sample):
        """
        """
        if isinstance(sample, np.ndarray):
            sample = sample.flatten()

        elif isinstance(sample, dict) or isinstance(sample, tuple):
            sample = self._flatten_composite_sample(sample)

        elif isinstance(sample, numbers.Number):
            sample = np.array((sample,))

        else:
            msg  = "ERROR: Unknown sample type of {type(sample)} "
            msg += "encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

        return sample

    def _flatten_composite_sample(self, sample):
        """
        """
        # TODO: this could be memory intesive with very large
        # spaces. We might want to pu effort into tracking the
        # expected sizes of each sub-space.
        flattened_data = []

        if isinstance(sample, dict):
            for key, item in sample.items():
                data = self.flatten_sample(item)
                flattened_data.append(data)

        elif isinstance(sample, tuple):
            for item in sample:
                data = self.flatten_sample(item)
                flattened_data.append(data)

        else:
            msg  = "ERROR: Unknown composite sample type of {type(sample)} "
            msg += "encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

        return np.concatenate(flattened_data)

    def _is_composite_space(self, space):
        """
        """
        return type(space) in [Dict, Tuple, Sequence, Graph]


class FlatteningTuple(Tuple, FlatteningCompositeSpace):
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

        old_gym_spaces = [\
            old_gym.spaces.Box,
            old_gym.spaces.Discrete,
            old_gym.spaces.MultiDiscrete,
            old_gym.spaces.MultiBinary,
            old_gym.spaces.Tuple,
            old_gym.spaces.Dict]

        for i in range(len(sub_spaces)):
            space = sub_spaces[i]

            if type(space) in old_gym_spaces:
                space = gym_space_to_gymnasium_space(space)
                sub_spaces[i] = space

            if type(space) not in self.supported_spaces:
                msg  = f"ERROR: sub space {space} is not currently supported by "
                msg += f"the FlatteningTuple. Supported sub-spaces are "
                msg += f"{self.supported_spaces}."
                rank_print(msg)
                comm.Abort()

            #
            # TODO: we could probably support multi-dimensional sub-spaces when
            # space.is_np_flattenable evaluates to True.
            #
            if len(space.shape) > 1 and not self.is_np_flattenable:
                msg  = f"ERROR: FlatteningTuple encountered space, {space}, that is "
                msg += f"not flattenable."
                rank_print(msg)
                comm.Abort()

            sample = self.flatten_sample(space.sample())

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
        data = super().sample()
        if self.auto_flatten:
            return self.flatten_sample(data)
        return data

    @property
    def shape(self):
        return (self.flattened_size,)


class FlatteningDict(Dict, FlatteningCompositeSpace):
    """
    A wrapper around a gymnasium Dict space that allows us
    to get combined/flattened samples.
    """

    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__(*args, **kw_args)

        self.sample_sizes   = []

        old_gym_spaces = [\
            old_gym.spaces.Box,
            old_gym.spaces.Discrete,
            old_gym.spaces.MultiDiscrete,
            old_gym.spaces.MultiBinary,
            old_gym.spaces.Tuple,
            old_gym.spaces.Dict]

        for key in self.spaces:
            space = self.spaces[key]

            if type(space) in old_gym_spaces:
                space = gym_space_to_gymnasium_space(space)
                self.spaces[key] = space

            if type(space) not in self.supported_spaces:
                msg  = f"ERROR: sub space {space} is not currently supported by "
                msg += f"the FlatteningDict. Supported sub-spaces are "
                msg += f"{self.supported_spaces}."
                rank_print(msg)
                comm.Abort()

            if len(space.shape) > 1 and not self.is_np_flattenable:
                msg  = f"ERROR: FlatteningDict encountered space, {space}, that is "
                msg += f"not flattenable."
                rank_print(msg)
                comm.Abort()

            sample = self.flatten_sample(space.sample())

            if type(sample) == np.ndarray:
                self.sample_sizes.append(sample.size)
            else:
                self.sample_sizes.append(1)

        self.sample_sizes   = np.array(self.sample_sizes, dtype=np.int32)
        self.flattened_size = self.sample_sizes.sum()

    def sample(self):
        """
        Sample the space.
        """
        data = super().sample()
        if self.auto_flatten:
            return self.flatten_sample(data)
        return data

    @property
    def shape(self):
        return (self.flattened_size,)


class SparseFlatteningCompositeSpace(FlatteningCompositeSpace):

    def __init__(self, *args, **kw_args):
        """
        """
        self.is_sparse    = False
        self.sparse_space = None

    def _sparsify_tuple_space(self, tuple_space):
        """
        """
        sparse_spaces = []
        sparse_idxs   = []
        for s_idx, space in enumerate(tuple_space):
            if type(space) in self.supported_spaces:

                if self._is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces.append(space)
                    sparse_idxs.append(sparse_idxs)
            else:
                self.is_sparse = True
                msg  = f"WARNING: encountered a Tuple space containing an unsupported "
                msg += f"space type of {type(space)}. It will be ignored when "
                msg += f"flattening."
                rank_print(msg)

        return sparse_idxs, Tuple(sparse_spaces)

    def _sparsify_dict_space(self, dict_space):
        """
        """
        sparse_spaces = {}
        for key, space in dict_space.items():
            if type(space) in self.supported_spaces:

                if self._is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces[key] = space
            else:
                self.is_sparse = True
                msg  = f"WARNING: spaces key {key} maps to an unsupported "
                msg += f"space type of {type(space)}. It will be ignored when "
                msg += f"flattening."
                rank_print(msg)

        return Dict(sparse_spaces)

    def _sparsify_composite_space(self, composite):
        """
        """
        if isinstance(composite, Dict):

            #
            # NOTE: the SparseFlatteningDict will sparsify itself during
            # construction.
            #
            sparse_dict = SparseFlatteningDict(
                spaces = composite.spaces,
                seed   = composite._np_random)

            return sparse_dict

        elif isinstance(composite, Tuple):

            #
            # NOTE: the SparseFlatteningTuple will sparsify itself during
            # construction.
            #
            sparse_tuple = SparseFlatteningTuple(
                spaces = composite.spaces,
                seed   = composite._np_random)

            return sparse_tuple

        self.is_sparse = True
        msg  = f"WARNING: encountered a Tuple space containing an unsupported "
        msg += f"space type of {type(space)}. It will be ignored when "
        msg += f"flattening."
        rank_print(msg)
        return None

    def sparse_sample(self, *args, **kw_args):
        """
        """
        return self.sparse_space.sample(*args, **kw_args)

    def _flatten_sample(self, sample):
        """
        """
        if isinstance(sample, np.ndarray):
            sample = sample.flatten()

        elif isinstance(sample, dict) or isinstance(sample, tuple):
            sample = self._flatten_composite_sample(sample)

        elif isinstance(sample, numbers.Number):
            sample = np.array((sample,))

        else:
            msg  = "ERROR: Unknown sample type of {type(sample)} "
            msg += "encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

        return sample

    def _flatten_composite_sample(self, sample):
        """
        """
        flattened_data = []

        if isinstance(sample, dict):
            for key, item in sample.items():
                data = self._flatten_sample(item)
                flattened_data.append(data)

        elif isinstance(sample, tuple):
            for item in sample:
                data = self._flatten_sample(item)
                flattened_data.append(data)

        else:
            msg  = "ERROR: Unknown composite sample type of {type(sample)} "
            msg += "encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

        return np.concatenate(flattened_data)

    def _sparsify_sample(self, sparse_space, dense_sample):
        """
        """
        if isinstance(sparse_space, SparseFlatteningTuple):
            sparse_sample = []
            for idx in sparse_space.sparse_idxs:
                sparse_sample.append(dense_sample[idx])

            return tuple(sparse_sample)

        elif isinstance(sparse_space, SparseFlatteningDict):

            sparse_sample = {}
            for key in sparse_space.keys():
                sparse_sample[key] = self._sparsify_sample(sparse_space[key], dense_sample[key])

            return dict(sparse_sample)

        elif isinstance(dense_sample, np.ndarray):
            return dense_sample

        elif isinstance(dense_sample, numbers.Number):
            return np.array([dense_sample])

        else:
            msg  = "ERROR: Unknown sample from space {type(sparse_space)} "
            msg += "encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

    def _sparse_flatten_sample(self, sparse_space, dense_sample):
        """
        """
        if sparse_space.is_sparse:
            sparse_sample = self._sparsify_sample(sparse_space, dense_sample)
        else:
            sparse_sample = dense_sample

        return self._flatten_composite_sample(sparse_sample)

    def sparse_flatten_sample(self, dense_sample):
        """
        """
        sparse_sample = self._sparse_flatten_sample(self.sparse_space, dense_sample)
        return self._flatten_composite_sample(sparse_sample)

    def sample(self):
        """
        Sample the space.
        """
        data = super().sample()
        if self.auto_flatten:
            return self.sparse_flatten_sample(data)
        return data


class SparseFlatteningTuple(FlatteningTuple, SparseFlatteningCompositeSpace):
    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__(*args, **kw_args)

        self.is_sparse = False
        self.sparse_idxs, sparse_spaces = self._sparsify_tuple_space(Tuple(self.spaces))

        if self.is_sparse:
            self.sparse_spaces = self.sparse_spaces
        else:
            self.sparse_spaces = self.spaces


class SparseFlatteningDict(FlatteningDict, SparseFlatteningCompositeSpace):

    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__(*args, **kw_args)

        self.is_sparse = False
        sparse_spaces  = self._sparsify_dict_space(Dict(self.spaces))

        if self.is_sparse:
            self.sparse_spaces = self.sparse_spaces
        else:
            self.sparse_spaces = self.spaces


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

    # FIXME: should we instead try to override the shape method like we do
    # for the FlatteningTuple?
    #@property
    #def shape(self):
    #    return (1,)
