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
    Validate the observation space of the given environment, and replace
    any spaces/sub-spaces that we need to.

    Parameters:
    -----------
    env: environment
        The environment to inspect.

    Returns:
    --------
    The input environment with an (possibly) updated observation space.
    """
    is_discrete_space = lambda s : type(s) == Discrete or type(s) == old_gym.spaces.Discrete
    get_space_args    = lambda s : (s.n, s.start, s._np_random)

    if is_discrete_space(env.observation_space):
        n, start, seed = get_space_args(env.observation_space)
        env.observation_space = ShapelyDiscrete(n = n, start = start, seed = seed)

    elif isinstance(env.observation_space, Tuple):
        new_spaces = []
        for i in range(len(env.observation_space)):
            if is_discrete_space(env.observation_space[i]):
                n, start, seed = get_space_args(env.observation_space[i])
                new_spaces.append(ShapelyDiscrete(n = n, start = start, seed = seed))
            else:
                new_spaces.append(env.observation_space[i])

        env.observation_space = SparseFlatteningTuple(
            new_spaces, env.observation_space._np_random)
        env.observation_space.auto_flatten = False

    elif type(env.observation_space) == Dict:
        for key in env.observation_space:
            if is_discrete_space(env.observation_space[key]):
                n, start, seed = get_space_args(env.observation_space[key])
                env.observation_space[key] = ShapelyDiscrete(n = n, start = start, seed = seed)

        env.observation_space = SparseFlatteningDict(
            env.observation_space.spaces, env.observation_space._np_random)
        env.observation_space.auto_flatten = False

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
    if isinstance(space, FlatteningCompositeSpace):
        return space

    if isinstance(space, old_gym.spaces.Box):
        space = gym.spaces.Box(
            low   = space.low,
            high  = space.high,
            shape = space.shape,
            dtype = space.dtype)

    elif isinstance(space, old_gym.spaces.Discrete):
        try:
            space = gym.spaces.Discrete(
                n     = space.n,
                start = space.start)
        except:
            space = gym.spaces.Discrete(
                n = space.n)

    elif isinstance(space, old_gym.spaces.MultiBinary):
        space = gym.spaces.MultiBinary(
            n = space.n)

    elif isinstance(space, old_gym.spaces.MultiDiscrete):
        space = gym.spaces.MultiDiscrete(
            nvec  = space.nvec,
            dtype = space.dtype)

    elif isinstance(space, old_gym.spaces.Dict):
        new_space = gym.spaces.Dict()

        for key in space:
            new_space[key] = gym_space_to_gymnasium_space(space[key])

        space = new_space

    elif isinstance(space, old_gym.spaces.Tuple):
        new_space = []

        for subspace in space:
            new_space.append(gym_space_to_gymnasium_space(subspace))

        space = gym.spaces.Tuple(new_space)

    return space


def is_composite_space(space):
    """
    Is the given space a composite space?

    Parameters:
    -----------
    space: gymnasium.spaces.Space
        The space to check

    Returns:
    --------
    True iff the space is composite.
    """
    if (isinstance(space, Dict) or
        isinstance(space, Tuple) or
        isinstance(space, Sequence) or
        isinstance(space, Graph)):
        return True
    return False


class FlatteningCompositeSpace(ABC):
    """
    An abstract base class for creating composite spaces
    that can flatten their samples.
    """

    def __init__(self, *args, **kw_args):
        super().__init__()

        self._auto_flatten   = True
        self._flattened_size = None

    @abstractmethod
    def sample(self):
        return

    def _space_is_supported(self, space):
        """
        Is the given spaces supported?

        Parameters:
        -----------
        space: Space
            The space to check.

        Returns:
        --------
        True iff the given space is supported.
        """
        for supp_space in self.supported_spaces:
            if isinstance(space, supp_space):
                return True
        return False

    def flatten_sample(self, sample):
        """
        Attempt to flatten the sample.

        Parameters:
        -----------
        sample: Any
            The sample to flatten.

        Returns:
        --------
        A flattened version of the sample as a np.ndarray.
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
        Attempt to flatten a composite sample.

        Parameters:
        -----------
        sample: dict or tuple
            The sample to flatten.

        Returns:
        --------
        A flattened version of the sample as a np.ndarray.
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

        if len(flattened_data) > 0:
            return np.concatenate(flattened_data)
        else:
            return np.zeros(0)

    def _wrap_sub_spaces(self, space):
        """
        Replace the subspaces of a composite space with flattening
        veresions of those spaces.

        Parameters:
        -----------
        space: Any, should be composite Space
            The space to wrap sub-spaces of. If it's not a Dict or Tuple,
            it will be a no-op.

        Returns:
        --------
        The input space with replaced subspaces.
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
        Replace composite spaces with flattening versions of those spaces.

        Parameters:
        -----------
        space: Any
            The space to wrap.

        Returns:
        --------
        A flattening version of the input space.
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
        Convert all input spaces to from gym to gymnasium.

        Parameters:
        -----------
        spaces: dict, Dict, or iterable
            A container of spaces.
        require_support: bool
            Should we throw an error if we encounter unsupported spaces?

        Returns:
        --------
        The input spaces with all gym version converted to gymnasium.
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
        Calculate the sample sizes for the given spaces.

        Parameters:
        -----------
        spaces: dict, Dict, or iterable
            The spaces to inspect

        Returns:
        --------
        A np.ndarray of the sample sizes for the input spaces.
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
        Update the flattened sizes of all of the given spaces.

        Parameters:
        -----------
        spaces: iterable of spaces
            The spaces to update the flattened size of.
        """
        for space in spaces:
            if isinstance(space, FlatteningCompositeSpace):
                space._update_flattened_size()

    def _update_flattened_size(self):
        """
        Update the flattened size for this space.
        """
        temp = self._auto_flatten
        self._auto_flatten   = True
        self._flattened_size = self.sample().size
        self._auto_flatten   = temp
        self._update_flattened_sizes(self.spaces)

    @property
    def flattened_size(self):
        return self._flattened_size

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
        super().__init__(spaces, *args, **kw_args)

        self.sample_sizes = self._calculate_sample_sizes(spaces)
        self._update_flattened_size()

    def sample(self):
        """
        Sample the space.
        """
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
        data = super().sample()
        if self._auto_flatten:
            return self.flatten_sample(data)
        return data

    @property
    def shape(self):
        return (self._flattened_size,)


class SparseFlatteningCompositeSpace(FlatteningCompositeSpace):
    """
    A wrapper around a gymnasium composite space that allows us
    to get combined/flattened samples and ignore unsupported sub-spaces.
    """

    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        self._is_sparse    = False
        self._sparse_space = None
        self._dense_space  = None
        self._mode         = "sparse"

    def _sparsify_tuple_space(self, tuple_space):
        """
        Recursively remove unsupported sub-spaces from the
        given Tuple space.

        Parameters:
        -----------
        tuple_space: gymnasium Tuple
             The tuple space to sparsify.

        Returns:
        --------
        A version of the Tuple space that has unsupported sub-spaces removed.
        """
        sparse_spaces = []
        sparse_idxs   = []
        for s_idx, space in enumerate(tuple_space):
            if self._space_is_supported(space):

                if is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces.append(space)
                    sparse_idxs.append(s_idx)
            else:
                self._is_sparse = True

                msg  = f"\nWARNING: encountered a Tuple space containing an unsupported "
                msg += f"space type of {type(space)}. It will be ignored when "
                msg += f"flattening.\n"
                sys.stderr.write(msg)

        return np.array(sparse_idxs), Tuple(sparse_spaces)

    def _sparsify_dict_space(self, dict_space):
        """
        Recursively remove unsupported sub-spaces from the
        given Dict space.

        Parameters:
        -----------
        dict_space: gymnasium Dict
             The dict space to sparsify.

        Returns:
        --------
        A version of the Dict space that has unsupported sub-spaces removed.
        """
        sparse_spaces = {}
        for s_idx, key in enumerate(dict_space):
            space = dict_space[key]

            if self._space_is_supported(space):

                if is_composite_space(space):
                    space = self._sparsify_composite_space(space)

                if space is not None:
                    sparse_spaces[key] = space

            else:
                self._is_sparse = True

                msg  = f"\nWARNING: encountered a Tuple space containing an unsupported "
                msg += f"space type of {type(space)}. It will be ignored when "
                msg += f"flattening.\n"
                sys.stderr.write(msg)

        return Dict(sparse_spaces)

    def _sparsify_composite_space(self, composite):
        """
        Remove unsupported sub-spaces from the given composite space.

        Parameters:
        -----------
        composite: A composite space
            The space to sparsify.

        Returns:
        --------
        A version of the composite space with unsupported sub-spaces removed.
        """
        if isinstance(composite, Dict):
            sparse_dict = self._sparsify_dict_space(composite)
            return sparse_dict

        elif isinstance(composite, Tuple):
            _, sparse_tuple = self._sparsify_tuple_space(composite)
            return sparse_tuple

        self._is_sparse = True
        msg  = f"\nWARNING: encountered a composite space containing an unsupported "
        msg += f"space type of {type(space)}. It will be ignored when "
        msg += f"flattening."
        sys.stderr.write(msg)
        return None

    def sparse_sample(self, *args, **kw_args):
        """
        Sample the sparse version of our space.

        Returns:
        --------
        The sampled tuple.
        """
        return self._sparse_space.sample(*args, **kw_args)

    def _sparsify_sample(self, space, dense_sample):
        """
        Given a dense sample, remove the sub-samples that come from
        unsupported sub-spaces.

        Parameters:
        -----------
        space: gymnasium space
            The space that was sampled.
        dense_sample: Any
            The dense sample to sparsify.

        Returns:
        --------
        A sparsified version of the dense sample.
        """
        if isinstance(space, SparseFlatteningTuple):
            sparse_sample = []
            start_idx     = 0
            for s_idx in space.sparse_idxs:
                sparse_sample.append(
                    self._sparsify_sample(
                        space._dense_space.spaces[s_idx], dense_sample[s_idx]))


            return tuple(sparse_sample)

        elif isinstance(space, SparseFlatteningDict):

            sparse_sample = {}
            for key in space._sparse_space.keys():
                sparse_sample[key] = self._sparsify_sample(space._dense_space.spaces[key], dense_sample[key])

            return dict(sparse_sample)

        elif isinstance(space, Tuple) or isinstance(space, Dict):
            return dense_sample

        elif isinstance(dense_sample, np.ndarray):
            return dense_sample

        elif isinstance(dense_sample, numbers.Number):
            return np.array([dense_sample])

        else:
            msg  = f"Unknown sample from space {type(space)} of type {type(dense_sample)} "
            msg += f"encountered! Bailing..."
            raise ValueError(msg)

    def _sparse_flatten_sample(self, space, dense_sample):
        """
        Given a dense sample, remove sub-samples from unsupported sub-spaces,
        and then flatten the sample.

        Parameters:
        -----------
        space: gymnasium space
            The space that was sampled.
        dense_sample: Any
            The dense sample to sparsify.

        Returns:
        --------
        A sparsified and flattened version of the dense sample.
        """
        if self._is_sparse:
            sparse_sample = self._sparsify_sample(space, dense_sample)
        else:
            sparse_sample = dense_sample

        return self.flatten_sample(sparse_sample)

    def sparse_flatten_sample(self, dense_sample):
        """
        Given a dense sample, remove sub-samples from unsupported sub-spaces,
        and then flatten the sample.

        Parameters:
        -----------
        dense_sample: Any
            The dense sample to sparsify.

        Returns:
        --------
        A sparsified and flattened version of the dense sample.
        """
        if isinstance(dense_sample, np.ndarray):
            return dense_sample
        return self._sparse_flatten_sample(self, dense_sample)

    def _wrap_space(self, space):
        """
        Convert the given space to a sparse flattening version of
        itself, if available.

        Parameters:
        -----------
        space: gymnasium space
            The space to wrap.

        Returns:
        --------
        If the given space is a supported composite space, a sparse
        flattening version of it will be returned. Otherwise, the original
        space is returned.
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
        Update the flattened sizes of all sparse flattening composite
        sub-spaces in the given spaces.

        Parameters:
        -----------
        spaces: iterable
            An iterable of spaces.

        Returns:
        --------
        The input spaces.
        """
        for space in spaces:
            if isinstance(space, SparseFlatteningCompositeSpace):
                space._update_flattened_size()

    def _update_flattened_size(self):
        """
        Update flattened size for all sub-spaces.
        """
        temp       = self._mode
        self.mode  = "sparse"
        self._flattened_size = self.sample().size
        self._update_flattened_sizes(self._dense_space.spaces)
        self._update_flattened_sizes(self._sparse_space.spaces)
        self.mode  = temp

    def _get_space_tree_strings(self, spaces, str_tree, space_type, indent):
        """
        Construct a string tree of the given spaces.

        Paramters:
        ----------
        spaces: iterable
            The spaces to construct a string tree of.
        str_tree: str
            An existing string tree to add the new one to.
        space_type: str
            The type of the spaces we're looking at (dense, sparse, etc.).
        indent: str
            Any indentation that should be used when constructing the tree.

        Returns:
        --------
        A potentially updated str_tree.
        """
        if isinstance(spaces, dict) or isinstance(spaces, Dict):
            space_iter = spaces.keys()
        else:
            space_iter = iter(range(len(spaces)))

        for iter_i in space_iter:
            if isinstance(spaces[iter_i], SparseFlatteningCompositeSpace):
                str_tree = spaces[iter_i].get_tree_str(str_tree, space_type, indent)

        return str_tree

    def get_tree_str(self, str_tree="", space_type="", indent=""):
        """
        Construct a string tree of ourself.

        Paramters:
        ----------
        str_tree: str
            An existing string tree to add the new one to.
        space_type: str
            The type of the spaces we're looking at (dense, sparse, etc.).
        indent: str
            Any indentation that should be used when constructing the tree.

        Returns:
        --------
        A string tree representation of ourself.
        """
        str_tree = f"\n{str_tree}{indent}{type(self)} {self._mode} mode ({space_type})\n"
        str_tree = self._get_space_tree_strings(self.spaces, str_tree, "spaces", indent + "    ")
        str_tree = self._get_space_tree_strings(self._dense_space.spaces, str_tree, "dense_spaces", indent + "    ")
        str_tree = self._get_space_tree_strings(self._sparse_space.spaces, str_tree, "sparse_spaces", indent + "    ")
        return str_tree

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
        Update the modes of all the given spaces.

        Parameters:
        -----------
        spaces: iterable
            An iterable of spaces.
        mode: str
            The mode to switch to.

        Returns:
        --------
        The input spaces with their mode updated.
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
    """
    The SparseFlatteningTuple allows for two modes:
        1. dense: the space will act the same as a Tuple space,
           but the user can "sparsify" and flatten dense samples, which removes all
           sub-samples that come from unsupported sub-spaces before flattening.
           The dense samples that come from calling sample() must be sent
           through the "sparse_flatten_sample" method to achieve this.
        2. sparse: when sampling, resulting samples will not contain
           any sub-samples that would have come from un-supported sub-spaces.
    """

    def __init__(self, spaces, *args, **kw_args):
        """
        Parameters:
        -----------
        spaces: iterable
            The spaces to utilize in our Tuple space.
        """
        spaces = self._convert_spaces_to_gymnasium(spaces, require_supported=False)

        SparseFlatteningCompositeSpace.__init__(self, *args, **kw_args)
        super().__init__(spaces, *args, **kw_args)

        self.spaces = self._wrap_sub_spaces(Tuple(self.spaces)).spaces

        self._is_sparse = False
        self.sparse_idxs, sparse_space = self._sparsify_tuple_space(Tuple(self.spaces))
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
    """
    The SparseFlatteningDict allows for two modes:
        1. dense: the space will act the same as a Dict space,
           but the user can "sparsify" and flatten dense samples, which removes all
           sub-samples that come from unsupported sub-spaces before flattening.
           The dense samples that come from calling sample() must be sent
           through the "sparse_flatten_sample" method to achieve this.
        2. sparse: when sampling, resulting samples will not contain
           any sub-samples that would have come from un-supported sub-spaces.
    """

    def __init__(self, spaces, *args, **kw_args):
        """
        Parameters:
        -----------
        spaces: dict
            The spaces to utilize in our Dict space.
        """
        spaces = self._convert_spaces_to_gymnasium(spaces, require_supported=False)

        SparseFlatteningCompositeSpace.__init__(self, spaces, *args, **kw_args)
        super().__init__(Dict(spaces), *args, **kw_args)

        self.spaces = self._wrap_sub_spaces(Dict(self.spaces)).spaces

        self._is_sparse   = False
        sparse_space = self._sparsify_dict_space(Dict(spaces))
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
    """
    A version of Discrete that has shape (1,) and returns
    samples in a numpy array.
    """
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)

    def sample(self, *args, **kw_args):
        return np.array((super().sample(*args, **kw_args),))

    @property
    def shape(self):
        return (1,)

