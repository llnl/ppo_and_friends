"""
    Wrappers that are used to allow an environment built for version
    X to work in gym version Y.
"""
import gymnasium as gym
import numpy as np
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.spaces import gym_space_to_gymnasium_space

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class Gym21ToGymnasium():
    """
       There are a lot of environments that are still only supported
       in version 0.21 of gym. Gymnasium looks a lot more like 0.26,
       but the spaces also need to be converted.
    """
    def __init__(self, env):
        """
            Initialize the wrapper.
        """
        self.env               = env
        self.observation_space = gym_space_to_gymnasium_space(env.observation_space)
        self.action_space      = gym_space_to_gymnasium_space(env.action_space)
        self.ale               = getattr(env, "ale", None)
        self.spec              = getattr(env, "spec", None)

    def seed(self, seed):
        """
            Set the seed. This changed in .26.

            Arguments:
                seed    The seed to set.
        """
        self.env.seed(seed)

    def step(self, actions, **kw_args):
        """
            Step through the environment.

            Arguments:
                actions    The actions to take.

            Returns:
                A tuple containing obs, reward, terminated, truncated, 
                and info.
        """
        obs, reward, terminated, info = self.env.step(actions)

        truncated = np.zeros(len(terminated)).astype(bool)

        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kw_args):
        """
            Reset the environment.

            Returns:
                The obs and info.
        """
        obs = self.env.reset()
        return obs, {}

    def render(self, *args, **kw_args):
        """
            Render the environment.
        """
        return self.env.render(*args, **kw_args)

