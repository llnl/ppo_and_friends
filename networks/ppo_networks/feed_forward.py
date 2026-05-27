import torch
from torch import nn
import numpy as np
from ppo_and_friends.networks.utils import *
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.networks.ppo_networks.base import PPONetwork, SingleSplitObservationNetwork

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class FeedForwardNetwork(PPONetwork):

    def __init__(self,
                 in_shape,
                 out_shape,
                 out_init     = None,
                 activation   = nn.ReLU(),
                 hidden_size  = 128,
                 hidden_depth = 3,
                 is_embedded  = False,
                 **kw_args):
        """
            A class defining a customizable feed-forward network.

            Arguments:
                out_shape       The shape of the output.
                out_init        A std weight to apply to the output layer.
                activation      The activation function to use on the output
                                of hidden layers.
                hidden_size     Can either be an int or list of ints. If an int,
                                all layers will be this size. Otherwise, a list
                                designates the size for each layer. Note that
                                the hidden_depth argument is ignored if this
                                argument is a list and the depth is instead
                                taken from the length of the list. Note that
                                this argument can be set to 0 or an empty list,
                                resulting in only an input and output layer.
                hidden_depth    The number of hidden layers. Note that this is
                                ignored if hidden_size is a list.
                is_embedded     If True, this network will be treated as being
                                embedded in a large network, and the output
                                from the output layer will be sent through
                                the activation function.
        """

        super(FeedForwardNetwork, self).__init__(
            in_shape  = in_shape,
            out_shape = out_shape,
            **kw_args)

        self.is_embedded = is_embedded
        self.activation  = activation

        self.sequential_net = \
            create_sequential_network(
                in_size      = self.in_size,
                out_size     = self.out_size,
                hidden_size  = hidden_size,
                hidden_depth = hidden_depth,
                activation   = activation,
                out_init     = out_init)

    def forward(self, _input):
        out = _input.flatten(start_dim=1)
        out_ = self.sequential_net(out)

        if torch.isnan(out_).any():
            msg = "ERROR: self.sequential_net as output nan values! "
            msg += "\n{out}"
            rank_print(msg)
            comm.Abort()

        #
        # If this network is embedded in a larger network,
        # we want to treat it as such and return the output
        # after activation.
        #
        if self.is_embedded:
            return self.activation(out_)

        out = self.output_func(out_)

        return self._shape_output(out)


class SplitObsNetwork(SingleSplitObservationNetwork):
    def __init__(self,
                 in_shape,
                 out_shape,
                 out_init,
                 left_hidden_size      = 64,
                 left_hidden_depth     = 3,
                 left_out_size         = 64,
                 right_hidden_size     = 64,
                 right_hidden_depth    = 3,
                 right_out_size        = 64,
                 combined_hidden_size  = 128,
                 combined_hidden_depth = 2,
                 activation            = nn.ReLU(),
                 **kw_args):
        """
            A class defining a customizable "split" network, where we
            split the network into 2 halves before joining them into
            a "merged" section. The idea here comes from arXiv:1610.05182v1,
            although this implementation more closely resembles the network
            used in arXiv:1707.02286v2. In both cases, the goal is to
            split proprioceptive/egocentric information (think joints,
            angular velocities, etc.) from exteroceptive information that
            relates to the environment (terrain sensors, position in relation
            to the terrain, etc.).

            Arguments:
             out_shape             The expected shape for the output. For
                                   instance, if the expected output shape is
                                   (batch_size, 16, 4), out_shape would be
                                   (16, 4).
             out_init              A std weight to apply to the output layer.
             left_hidden_size      The number of output neurons for each hidden
                                   layer of the left network. Note that this can
                                   be set to 0, resulting in only an input and
                                   output layer.
             left_hidden_depth     The number of hidden layers in the left
                                   network.
             left_out_size         The number of output neurons for the left
                                   network.
             right_hidden_size     The number of output neurons for each hidden
                                   layer of the right network.
             right_hidden_depth    The number of hidden layers in the right
                                   network.
             right_out_size        The number of output neurons for the right
                                   network.

             combined_hidden_size  The number of output neurons for each hidden
                                   layer of the right network. Note that this
                                   can be set to 0, resulting in only an input
                                   and output layer.
             combined_hidden_depth The number of hidden layers in the right
                                   network.
             activation            The activation function to use on the output
                                   of hidden layers.
        """


        super(SplitObsNetwork, self).__init__(
            **kw_args)

        self.activation = activation

        side_1_size = self.split_start
        side_2_size = self.in_size - self.split_start

        # TODO: in the orignal paper, there is a "low level" section and
        # a "high level" section. The low level section handles
        # proprioceptive information (joints, positions, etc.), and the
        # high level section sees everything but only sends a signal every
        # K iterations. A later paper uses a similar technique, but it's
        # unclear if they keep the "K iteration" approach (they're very
        # vague). At any rate, this later approach is slighty different in
        # that it splits between proprioceptive and exteroceptive (sensory
        # information about the environment).
        #

        #
        # Left side network.
        #
        s1_kw_args = kw_args.copy()
        s1_kw_args["name"] = self.name + "_s1"

        self.s1_net = FeedForwardNetwork(
            in_size      = side_1_size,
            hidden_size  = left_hidden_size,
            hidden_depth = left_hidden_depth,
            out_shape    = left_out_size,
            activation   = self.activation,
            is_embedded  = True,
            **s1_kw_args)

        #
        # Right side network.
        #
        s2_kw_args = kw_args.copy()
        s2_kw_args["name"] = self.name + "_s2"

        self.s2_net = FeedForwardNetwork(
            in_size      = side_2_size,
            hidden_size  = right_hidden_size,
            hidden_depth = right_hidden_depth,
            out_shape    = right_out_size,
            activation   = self.activation,
            is_embedded  = True,
            **s2_kw_args)

        #
        # Combined sides network.
        #
        combined_in_size = left_out_size + right_out_size
        c_kw_args = kw_args.copy()
        c_kw_args["name"] = self.name + "_combined"

        self.combined_layers = FeedForwardNetwork(
            in_size      = combined_in_size,
            hidden_size  = combined_hidden_size,
            hidden_depth = combined_hidden_depth,
            out_shape    = self.out_shape,
            activation   = self.activation,
            is_embedded  = False,
            out_init     = out_init,
            **c_kw_args)

    def forward(self, _input):
        out = _input.flatten(start_dim = 1)

        s1_out = out[:, 0 : self.split_start]
        s2_out = out[:, self.split_start : ]

        #
        # Side 1 (left side).
        #
        s1_out = self.s1_net(s1_out)

        #
        # Side 2 (right side).
        #
        s2_out = self.s2_net(s2_out)

        #
        # Full layers.
        #
        out = torch.cat((s1_out, s2_out), dim=1)
        out = self.combined_layers(out)

        out = self.output_func(out)

        return self._shape_output(out)
