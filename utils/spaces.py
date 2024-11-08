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

    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__()

        self.supported_spaces = [
            Discrete,
            MultiDiscrete,
            MultiBinary,
            Box,
            Dict,
            Tuple,
        ]

        self._auto_flatten = False

    def _space_is_supported(self, space):
        """
        """
        for supp_space in self.supported_spaces:
            if isinstance(space, supp_space):
                return True
        return False

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
            msg  = f"ERROR: Unknown sample type of {type(sample)} "
            msg += f"encountered! Bailing..."
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
            msg  = f"ERROR: Unknown composite sample type of {type(sample)} "
            msg += f"encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

        return np.concatenate(flattened_data)

    def _is_composite_space(self, space):
        """
        """
        return type(space) in [Dict, Tuple, Sequence, Graph]

    def _wrap_sub_spaces(self, space):
        """
        """
        if isinstance(space, Dict):

            for key in space:
                space[key] = self._wrap_space(space[key])

        elif isinstance(space, Tuple):

            wrapped_spaces = []
            for sub_space in space:
                wrapped_spaces.append(self._wrap_space(sub_space))

            space = Tuple(wrapped_spaces)

        return space

    def _wrap_space(self, space):
        """
        """
        if isinstance(space, Dict):

            for key in space:
                space[key] = self._wrap_sub_spaces(space[key])

            return SparseFlatteningDict(
                space.spaces, seed = space._np_random)

        elif isinstance(space, Tuple):

            wrapped_spaces = []
            for sub_space in space:
                wrapped_spaces.append(self._wrap_sub_spaces(sub_space))

            space = Tuple(wrapped_spaces)

            return SparseFlatteningTuple(
                space.spaces, seed = space._np_random)

        return space

    @property
    def auto_flatten(self):
        return self._auto_flatten

    @auto_flatten.setter
    def set_auto_flatten(self, auto_flatten):
        self._auto_flatten = auto_flatten


class FlatteningTuple(Tuple, FlatteningCompositeSpace):
    """
    A wrapper around a gymnasium Tuple space that allows us
    to get combined/flattened samples.
    """

    def __init__(self, spaces, *args, **kw_args):
        """
        Parameters:
        -----------
        spaces: iterable
            An iterable containing the sub-spaces to encapsulate.
        """
        #
        # Unfortunately, it looks like gymnasium spaces aren't respecting
        # the rules of multiple inheritance, which means we need to
        # call init manually...
        #
        FlatteningCompositeSpace.__init__(self, *args, **kw_args)

        self.sample_sizes   = []

        old_gym_spaces = [\
            old_gym.spaces.Box,
            old_gym.spaces.Discrete,
            old_gym.spaces.MultiDiscrete,
            old_gym.spaces.MultiBinary,
            old_gym.spaces.Tuple,
            old_gym.spaces.Dict]

        for i in range(len(spaces)):
            space = spaces[i]

            if type(space) in old_gym_spaces:
                space = gym_space_to_gymnasium_space(space)
                spaces[i] = space

            if not self._space_is_supported(space):
                msg  = f"WARNING: sub space {space} of type {type(space)} is not currently supported by "
                msg += f"the FlatteningTuple. Supported sub-spaces are "
                msg += f"{self.supported_spaces}."
                rank_print(msg)

            sample = self.flatten_sample(space.sample())

            if type(sample) == np.ndarray:
                self.sample_sizes.append(sample.size)
            else:
                self.sample_sizes.append(1)

        self.sample_sizes   = np.array(self.sample_sizes, dtype=np.int32)
        self.flattened_size = self.sample_sizes.sum()

        super().__init__(spaces, *args, **kw_args)

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
        #
        # Unfortunately, it looks like gymnasium spaces aren't respecting
        # the rules of multiple inheritance, which means we need to
        # call init manually...
        #
        FlatteningCompositeSpace.__init__(self, *args, **kw_args)

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
                
            if not self._space_is_supported(space):
                msg  = f"WARNING: sub space {space} of type {type(space)} is not currently supported by "
                msg += f"the FlatteningDict. Supported sub-spaces are "
                msg += f"{self.supported_spaces}."
                rank_print(msg)

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
        super().__init__(*args, **kw_args)
        self.is_sparse    = False
        self.sparse_space = None

    def _sparsify_tuple_space(self, tuple_space):
        """
        """
        sparse_spaces = []
        sparse_idxs   = []
        for s_idx, space in enumerate(tuple_space):
            if self._space_is_supported(space):

                if self._is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces.append(space)
                    sparse_idxs.append(s_idx)
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
            if self._space_is_supported(space):

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
            return self._sparsify_dict_space(composite)

        elif isinstance(composite, Tuple):
            _, sparse_tuple = self._sparsify_tuple_space(composite)
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

    def _sparsify_sample(self, space, dense_sample):
        """
        """
        if isinstance(space, SparseFlatteningTuple):
            sparse_sample = []
            for idx in space.sparse_idxs:
                sparse_sample.append(self._sparsify_sample(space.spaces[idx], dense_sample[idx]))

            return tuple(sparse_sample)

        elif isinstance(space, SparseFlatteningDict):

            sparse_sample = {}
            for key in space.sparse_space.keys():
                sparse_sample[key] = self._sparsify_sample(space.spaces[key], dense_sample[key])

            return dict(sparse_sample)

        elif isinstance(dense_sample, np.ndarray):
            return dense_sample

        elif isinstance(dense_sample, numbers.Number):
            return np.array([dense_sample])

        else:
            msg  = f"ERROR: Unknown sample from space {type(space)} of type {type(dense_sample)} "
            msg += f"encountered! Bailing..."
            rank_print(msg)
            comm.Abort()

    def _sparse_flatten_sample(self, space, dense_sample):
        """
        """
        if self.is_sparse:
            sparse_sample = self._sparsify_sample(space, dense_sample)
        else:
            sparse_sample = dense_sample

        return self.flatten_sample(sparse_sample)

    def sparse_flatten_sample(self, dense_sample):
        """
        """
        return self._sparse_flatten_sample(self, dense_sample)

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
        SparseFlatteningCompositeSpace.__init__(self, *args, **kw_args)

        self.spaces = self._wrap_sub_spaces(Tuple(self.spaces)).spaces

        self.is_sparse = False
        self.sparse_idxs, sparse_space = self._sparsify_tuple_space(Tuple(self.spaces))

        if self.is_sparse:
            self.sparse_space = sparse_space
        else:
            self.sparse_space = self


class SparseFlatteningDict(FlatteningDict, SparseFlatteningCompositeSpace):

    def __init__(self, spaces, *args, **kw_args):
        """
        """
        SparseFlatteningCompositeSpace.__init__(self, spaces, *args, **kw_args)
        super().__init__(Dict(spaces), *args, **kw_args)

        self.spaces = self._wrap_sub_spaces(Dict(self.spaces)).spaces

        self.is_sparse = False
        sparse_space  = self._sparsify_dict_space(Dict(spaces))

        if self.is_sparse:
            self.sparse_space = sparse_space
        else:
            self.sparse_space = self


class ShapelyDiscrete(Discrete):

    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__(*args, **kw_args)

    def sample(self, *args, **kw_args):
        """
        """
        return np.array((super().sample(*args, **kw_args),))

    @property
    def shape(self):
        return (1,)
