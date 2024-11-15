import gym as old_gym
import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict, Sequence, Graph
from gymnasium.spaces.space import Space
import numpy as np
import numbers
import sys
from abc import ABC, abstractmethod
import functools

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
        msg  = f"conversion of gym space {space} to "
        msg += f"gymnasium is not currently supported. Contact "
        msg += f"a developer to extend support for this space."
        raise TypeError(msg)

    return space

class FlatteningCompositeSpace(ABC):

    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__()

        self._auto_flatten   = True
        self._flattened_size = None

    @abstractmethod
    def sample(self):
        """
        """
        return

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

        elif isinstance(sample, str):
            sample = np.array((sample,))

        elif isinstance(sample, object):
            sample = np.array((sample,))

        else:
            msg  = f"Unknown sample type of {type(sample)} "
            msg += f"encountered! Bailing..."
            raise TypeError(msg)

        return sample

    def _flatten_composite_sample(self, sample):
        """
        """
        # TODO: this could be memory intesive with very large
        # spaces. We might want to put effort into tracking the
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
            msg  = f"Unknown composite sample type of {type(sample)} "
            msg += f"encountered! Bailing..."
            raise TypeError(msg)

        return np.concatenate(flattened_data)

    def _is_composite_space(self, space):
        """
        """
        if (isinstance(space, Dict) or
            isinstance(space, Tuple) or
            isinstance(space, Sequence) or
            isinstance(space, Graph)):
            return True
        return False

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

            return FlatteningDict(
                space.spaces, seed = space._np_random)

        elif isinstance(space, Tuple):

            wrapped_spaces = []
            for sub_space in space:
                wrapped_spaces.append(self._wrap_sub_spaces(sub_space))

            space = Tuple(wrapped_spaces)

            return FlatteningTuple(
                space.spaces, seed = space._np_random)

        return space

    def _convert_spaces_to_gymnasium(self, spaces, require_supported=True):
        """
        """
        old_gym_spaces = [\
            old_gym.spaces.Box,
            old_gym.spaces.Discrete,
            old_gym.spaces.MultiDiscrete,
            old_gym.spaces.MultiBinary,
            old_gym.spaces.Tuple,
            old_gym.spaces.Dict]

        if isinstance(spaces, dict) or isinstance(spaces, Dict):
            space_iter = spaces.keys()
        else:
            space_iter = iter(range(len(spaces)))

        for iter_i in space_iter:
            space = spaces[iter_i]

            if type(space) in old_gym_spaces:
                space = gym_space_to_gymnasium_space(space)
                spaces[iter_i] = space

            if not self._space_is_supported(space):
                msg  = f"\nWARNING: sub space {space} of type {type(space)} is not currently supported by "
                msg += f"the FlatteningCompositeSpace. Supported sub-spaces are "
                msg += f"{self.supported_spaces}.\n"

                if require_supported:
                    raise ValueError(msg)
                else:
                    sys.stderr.write(msg)

        return spaces

    def _calculate_sample_sizes(self, spaces):
        """
        """
        if isinstance(spaces, dict) or isinstance(spaces, Dict):
            space_iter = spaces.keys()
        else:
            space_iter = iter(range(len(spaces)))

        sample_sizes = []
        for iter_i in space_iter:
            space  = spaces[iter_i]
            sample = self.flatten_sample(space.sample())

            if type(sample) == np.ndarray:
                sample_sizes.append(sample.size)
            else:
                sample_sizes.append(1)

        return np.array(sample_sizes, dtype=np.int32)

    def _update_flattened_sizes(self, spaces):
        """
        """
        for space in spaces:
            if isinstance(space, FlatteningCompositeSpace):
                space._update_flattened_size()

    def _update_flattened_size(self):
        """
        """
        temp = self._auto_flatten
        self._auto_flatten   = True
        self._flattened_size = self.sample().size
        self._auto_flatten   = temp
        self._update_flattened_sizes(self.spaces)

    @property
    def auto_flatten(self):
        return self._auto_flatten

    @auto_flatten.setter
    def auto_flatten(self, auto_flatten):
        self._auto_flatten = auto_flatten

        if isinstance(self.spaces, dict) or isinstance(self.spaces, Dict):
            space_iter = self.spaces.keys()
        else:
            space_iter = iter(range(len(self.spaces)))

        for iter_i in space_iter:
            if isinstance(self.spaces[iter_i], FlatteningCompositeSpace):
                self.spaces[iter_i].auto_flatten = auto_flatten

    @property
    def shape(self):
        return (self._flattened_size,)

    @property
    def supported_spaces(self):
        return [
            Discrete,
            MultiDiscrete,
            MultiBinary,
            Box,
            Dict,
            Tuple,
        ]


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

        spaces = self._convert_spaces_to_gymnasium(spaces)
        self.sample_sizes = self._calculate_sample_sizes(spaces)
        self._update_flattened_size()

        super().__init__(spaces, *args, **kw_args)

    def sample(self):
        """
        Sample the space.
        """
        #print(f"CALLING FLATTENING TUPLE SAMPLE")#FIXME
        data = super().sample()
        if self._auto_flatten:
            return self.flatten_sample(data)
        return data

    @property
    def shape(self):
        return (self._flattened_size,)


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

        spaces = self._convert_spaces_to_gymnasium(spaces)
        self.sample_sizes = self._calculate_sample_sizes(spaces)
        self._update_flattened_size()

    def sample(self):
        """
        Sample the space.
        """
        #print(f"CALLING FLATTENING DICT SAMPLE")#FIXME
        data = super().sample()
        if self._auto_flatten:
            return self.flatten_sample(data)
        return data

    @property
    def shape(self):
        return (self._flattened_size,)


class SparseFlatteningCompositeSpace(FlatteningCompositeSpace):

    def __init__(self, *args, **kw_args):
        """
        """
        super().__init__(*args, **kw_args)
        self._is_sparse    = False
        self._sparse_space = None
        self._dense_space  = None
        self._mode         = "sparse"

    def _sparsify_tuple_space(self, tuple_space):
        """
        """
        sparse_spaces = []
        sparse_idxs   = np.zeros(len(tuple_space)).astype(np.int32)
        sparse_sizes  = np.zeros(len(tuple_space)).astype(np.int32)
        for s_idx, space in enumerate(tuple_space):
            if self._space_is_supported(space):

                if self._is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces.append(space)
                    sparse_idxs[s_idx] = 1

                    if self._is_composite_space(space):
                        sparse_sizes[s_idx] = 1

                    elif (isinstance(space, Box) or
                        isinstance(space, MultiDiscrete) or
                        isinstance(space, MultiBinary)):

                        size = functools.reduce(lambda a, b : a * b, space.shape)
                        sparse_sizes[s_idx] = size

                    elif isinstance(space, Discrete):
                        sparse_sizes[s_idx] = 1

                    else:
                        raise TypeError(f"Unknown space {space} of type {type(space)} encountered!")
            else:
                #
                # For unsupported spaces, we need to try to calculate its size.
                #
                sparse_sizes[s_idx] = self.flatten_sample(space.sample()).size
                self._is_sparse     = True

                msg  = f"\nWARNING: encountered a Tuple space containing an unsupported "
                msg += f"space type of {type(space)}. It will be ignored when "
                msg += f"flattening.\n"
                sys.stderr.write(msg)

        return sparse_idxs, sparse_sizes, Tuple(sparse_spaces)

    def _sparsify_dict_space(self, dict_space):
        """
        """
        sparse_spaces = {}
        sparse_idxs   = np.zeros(len(dict_space.keys())).astype(np.int32)
        sparse_sizes  = np.zeros(len(dict_space.keys())).astype(np.int32)
        for s_idx, key in enumerate(dict_space):
            space = dict_space[key]

            if self._space_is_supported(space):

                if self._is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces[key] = space
                    sparse_idxs[s_idx] = 1

                    if self._is_composite_space(space):
                        sparse_sizes[s_idx] = 1

                    elif (isinstance(space, Box) or
                        isinstance(space, MultiDiscrete) or
                        isinstance(space, MultiBinary)):

                        size = functools.reduce(lambda a, b : a * b, space.shape)
                        sparse_sizes[s_idx] = size

                    elif isinstance(space, Discrete):
                        sparse_sizes[s_idx] = 1

                    else:
                        raise TypeError(f"Unknown space {space} of type {type(space)} encountered!")
            else:
                #
                # For unsupported spaces, we need to try to calculate its size.
                #
                sparse_sizes[s_idx] = self.flatten_sample(space.sample()).size
                self._is_sparse     = True

                msg  = f"\nWARNING: encountered a Tuple space containing an unsupported "
                msg += f"space type of {type(space)}. It will be ignored when "
                msg += f"flattening.\n"
                sys.stderr.write(msg)

        return sparse_idxs, sparse_sizes, Dict(sparse_spaces)

    def _sparsify_composite_space(self, composite):
        """
        """
        if isinstance(composite, Dict):
            _, _, sparse_dict = self._sparsify_dict_space(composite)
            return sparse_dict

        elif isinstance(composite, Tuple):
            _, _, sparse_tuple = self._sparsify_tuple_space(composite)
            return sparse_tuple

        self._is_sparse = True
        msg  = f"\nWARNING: encountered a composite space containing an unsupported "
        msg += f"space type of {type(space)}. It will be ignored when "
        msg += f"flattening."
        sys.stderr.write(msg)
        return None

    def sparse_sample(self, *args, **kw_args):
        """
        """
        return self._sparse_space.sample(*args, **kw_args)

    def _sparsify_sample(self, space, dense_sample):
        """
        """
        ##print(f"\nSPARSIFYING SAMPLE {dense_sample}")#FIXME
        #print(f"SPACE: {space}")

        if isinstance(space, SparseFlatteningTuple):
            #print(f"SFT")#FIXME
            sparse_sample = []
            start_idx     = 0
            #print(f"SPACES SPARSE SIZES: {space.sparse_sizes}")
            for idx, keeper in enumerate(space.sparse_idxs):

                #stop_idx = start_idx + space.sparse_sizes[idx]
                #if keeper:
                #    print(f"KEEPING SAMPLE {idx} WITH SPACE {space.spaces[idx]}, SENDING DENSE RANGE {start_idx} : {stop_idx}")
                #    if space.sparse_sizes[idx] > 1:
                #        sparse_sample.append(
                #            self._sparsify_sample(
                #                space.spaces[idx], dense_sample[start_idx : stop_idx]))
                #    else:
                #        sparse_sample.append(
                #            self._sparsify_sample(
                #                space.spaces[idx], dense_sample[start_idx]))
                #start_idx = stop_idx

                if keeper:
                    #print(f"KEEPING SAMPLE {idx} WITH SPACE {space.spaces[idx]}, SENDING DENSE {dense_sample[idx]}")
                    sparse_sample.append(
                        self._sparsify_sample(
                            space._dense_space.spaces[idx], dense_sample[idx]))


            return tuple(sparse_sample)

        elif isinstance(space, SparseFlatteningDict):
            #print(f"SFD")#FIXME

            sparse_sample = {}
            for key in space._sparse_space.keys():
                #print(f"KEEPING SAMPLE {key} WITH SPACE {space.spaces[key]}, SENDING DENSE {dense_sample[key]}")
                sparse_sample[key] = self._sparsify_sample(space._dense_space.spaces[key], dense_sample[key])
            #start_idx     = 0
            #for idx, key in enumerate(space.spaces.keys()):
            #    stop_idx = space.sparse_sizes[idx]

            #    if space.sparse_idxs[idx]:
            #        print(f"KEEPING SAMPLE {idx} WITH SPACE {space.spaces[key]}, SENDING DENSE RANGE {start_idx} : {stop_idx}")
            #        sparse_sample[key] = self._sparsify_sample(space.spaces[key], dense_sample[start_idx : stop_idx])

            #    start_idx = start_idx + stop_idx

            return dict(sparse_sample)

        elif isinstance(space, Tuple) or isinstance(space, Dict):
            ##print(f"TUPLE")#FIXME
            return dense_sample

        elif isinstance(dense_sample, np.ndarray):
            #print(f"NDARRAY")#FIXME
            return dense_sample

        elif isinstance(dense_sample, numbers.Number):
            #print(f"NUMBER")#FIXME
            return np.array([dense_sample])

        else:
            msg  = f"Unknown sample from space {type(space)} of type {type(dense_sample)} "
            msg += f"encountered! Bailing..."
            raise ValueError(msg)

    def _sparse_flatten_sample(self, space, dense_sample):
        """
        """
        if self._is_sparse:
            sparse_sample = self._sparsify_sample(space, dense_sample)
        else:
            sparse_sample = dense_sample

        return self.flatten_sample(sparse_sample)

    def sparse_flatten_sample(self, dense_sample):
        """
        """
        #FIXME:
        #if isinstance(dense_sample, np.ndarray):
        #    return dense_sample

        return self._sparse_flatten_sample(self, dense_sample)

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

    def _update_flattened_sizes(self, spaces):
        """
        """
        for space in spaces:
            if isinstance(space, SparseFlatteningCompositeSpace):
                space._update_flattened_size()

    def _update_flattened_size(self):
        """
        """
        #FIXME: this is breaking things...
        temp       = self._mode
        #print(f"MODE IS {temp}")
        #print(f"CHANGING MODE FROM {temp} to sparse")#FIXME
        #self.mode  = "sparse"
        #self._flattened_size = self.sample().size

        ##self._update_flattened_sizes(self._dense_space.spaces)
        ##self._update_flattened_sizes(self._sparse_space.spaces)

        #print(f"SETTING BACK TO {temp}")#FIXME
        #self.mode  = temp

    #FIXME: remove after debugging
    def _get_modes(self, spaces, st, indent):
        """
        """

        if isinstance(spaces, dict) or isinstance(spaces, Dict):
            space_iter = spaces.keys()
        else:
            space_iter = iter(range(len(spaces)))

        for iter_i in space_iter:
            if isinstance(spaces[iter_i], SparseFlatteningCompositeSpace):
                spaces[iter_i].get_mode(st, indent)

    #FIXME: remove after debugging
    def get_mode(self, st="", indent=""):
        """
        """
        print(f"{indent}MODE: {self._mode} ({st})")#FIXME
        self._get_modes(self.spaces, "spaces", indent + "    ")
        self._get_modes(self._dense_space.spaces, "dense_spaces", indent + "    ")
        self._get_modes(self._sparse_space.spaces, "sparse_spaces", indent + "    ")

    @property
    def sparse_space(self):
        return self._sparse_space

    @property
    def dense_space(self):
        return self._dense_space

    @property
    def is_sparse(self):
        return self._is_sparse

    def _update_space_modes(self, spaces, mode):
        """
        """
        if isinstance(spaces, dict) or isinstance(spaces, Dict):
            space_iter = spaces.keys()
        else:
            space_iter = iter(range(len(spaces)))

        for iter_i in space_iter:
            if isinstance(spaces[iter_i], SparseFlatteningCompositeSpace):
                spaces[iter_i].mode = mode

        return spaces

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        assert mode in ["sparse", "dense"]

        self._mode                = mode
        self._sparse_space.spaces = self._update_space_modes(self._sparse_space.spaces, mode)
        self._dense_space.spaces  = self._update_space_modes(self._dense_space.spaces, mode)


class SparseFlatteningTuple(Tuple, SparseFlatteningCompositeSpace):

    def __init__(self, spaces, *args, **kw_args):
        """
        """
        spaces = self._convert_spaces_to_gymnasium(spaces, require_supported=False)

        SparseFlatteningCompositeSpace.__init__(self, *args, **kw_args)
        super().__init__(spaces, *args, **kw_args)

        self.spaces = self._wrap_sub_spaces(Tuple(self.spaces)).spaces

        self._is_sparse = False
        self.sparse_idxs, self.sparse_sizes, sparse_space = self._sparsify_tuple_space(Tuple(self.spaces))
        self._dense_space = Tuple(self.spaces)

        if self._is_sparse:
            self._sparse_space = sparse_space
        else:
            self._sparse_space = self._dense_space

        self._update_flattened_size()

    def sample(self):
        """
        Sample the space.
        """
        if self._mode == "sparse":
            return self.flatten_sample(self._sparse_space.sample())

        data = self._dense_space.sample()

        if self._auto_flatten:
            return self.sparse_flatten_sample(data)

        return data

    @property
    def shape(self):
        return (self._flattened_size,)


class SparseFlatteningDict(Dict, SparseFlatteningCompositeSpace):

    def __init__(self, spaces, *args, **kw_args):
        """
        """
        spaces = self._convert_spaces_to_gymnasium(spaces, require_supported=False)

        SparseFlatteningCompositeSpace.__init__(self, spaces, *args, **kw_args)
        super().__init__(Dict(spaces), *args, **kw_args)

        self.spaces = self._wrap_sub_spaces(Dict(self.spaces)).spaces

        self._is_sparse   = False
        self.sparse_idxs, self.sparse_sizes, sparse_space = self._sparsify_dict_space(Dict(spaces))
        self._dense_space = Dict(self.spaces)

        if self._is_sparse:
            self._sparse_space = sparse_space
        else:
            self._sparse_space = self._dense_space

        self._update_flattened_size()

    def sample(self):
        """
        Sample the space.
        """
        if self._mode == "sparse":
            return self.flatten_sample(self._sparse_space.sample())

        data = self._dense_space.sample()

        if self._auto_flatten:
            return self.sparse_flatten_sample(data)

        return data

    @property
    def shape(self):
        return (self._flattened_size,)


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
