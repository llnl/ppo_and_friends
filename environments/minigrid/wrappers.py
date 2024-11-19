"""
This module contains wrappers for gym's MiniGrid environments.
"""
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper, MultiAgentGymWrapper
from ppo_and_friends.utils.spaces import exchange_composite_obs_spaces

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class SingleAgentMiniGridWrapper(SingleAgentGymWrapper):

    def __init__(self,
                 env,
                 *args,
                 **kw_args):

        #
        # NOTE: minigrid uses Dict spaces for observations, and they often
        # use a custom observation within these dicts that contains commands.
        # We want to ignore these, so we utilize our SparseFlatteningComposite
        # spaces by replacing their obs space with a replica of our own.
        #
        super().__init__(exchange_composite_obs_spaces(env), *args, **kw_args)

    def _validate_obs(self,
                      obs):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: array-like or number or dict
            The agent observations.

        Returns:
        --------
        The agent observations.
        """
        if isinstance(obs, dict):
            obs = self.env.observation_space.sparse_flatten_sample(obs)

        obs = super()._validate_obs(obs)

        return obs

class MultiAgentMiniGridWrapper(MultiAgentGymWrapper):

    def __init__(self,
                 env,
                 *args,
                 **kw_args):

        #
        # NOTE: minigrid uses Dict spaces for observations, and they often
        # use a custom observation within these dicts that contains commands.
        # We want to ignore these, so we utilize our SparseFlatteningComposite
        # spaces by replacing their obs space with a replica of our own.
        #
        super().__init__(exchange_composite_obs_spaces(env), *args, **kw_args)

    def _validate_obs(self,
                      obs):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: dict
            The agent observations.

        Returns:
        --------
        The agent observations.
        """
        for i in range(self.num_agents):
            if isinstance(obs[i], dict):
                obs[i] = self.env.observation_space.sparse_flatten_sample(obs[i])

        obs = super()._validate_obs(obs)

        return obs
