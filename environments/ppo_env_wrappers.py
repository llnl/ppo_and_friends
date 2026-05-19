"""
    A home for generic environment wrappers. The should not be
    specific to any type of environment.
"""
import time
import numpy as np
import numbers
from copy import deepcopy
from functools import reduce
import sys
import os
from abc import ABC, abstractmethod
from ppo_and_friends.utils.mpi_utils import rank_print
from collections.abc import Iterable
from gymnasium.spaces import Dict, Tuple, Box, Discrete

try:
    import gym.spaces as old_gym_spaces
except ImportError:
    old_gym_spaces = None
import gymnasium as gym

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class IdentityWrapper(ABC):
    """
        A wrapper that acts exactly like the original environment but also
        has a few extra bells and whistles.
    """

    def __init__(self,
                 env,
                 test_mode = False,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env           The environment to wrap.
                test_mode     Are we in test mode? (bool)
        """
        super(IdentityWrapper, self).__init__(**kw_args)

        self.env               = env
        self.test_mode         = test_mode
        self.need_hard_reset   = True
        self.obs_cache         = None
        self.can_augment_obs   = False
        self.finalized         = False

        self.observation_space        = env.observation_space
        self.critic_observation_space = env.critic_observation_space
        self.action_space             = env.action_space
        self.null_actions             = env.null_actions

        self.agent_ids = tuple(agent_id for agent_id in
            self.action_space.keys())

        if (callable(getattr(self.env, "augment_observation", None)) and
            callable(getattr(self.env, "augment_critic_observation", None))):
            self.can_augment_obs = True

    def get_all_done(self):
        """
            Are all agents done?

            Returns:
                Whether or not all agents are done (bool).
        """
        return self.env.get_all_done()

    def get_num_agents(self):
        """
            Get the number of agents in this environment.

            Returns:
                The number of agents in the environment.
        """
        return len(self.observation_space.keys())

    def set_random_seed(self, seed):
        """
            Set the random seed for the environment.

            Arguments:
                seed    The seed value.
        """
        try:
            self.env.seed(seed)
        except:
            msg  = "WARNING: unable to set the environment seed. "
            msg += "You may witness stochastic behavior."
            rank_print(msg)
            pass

        for key in self.action_space:
            self.action_space[key].seed(seed)

    def _cache_step(self, action):
        """
            Take a single step in the environment using the given
            action, and cache the observation for soft resets.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, terminated, truncated,
                and info tuple.
        """
        obs, critic_obs, reward, terminated, truncated, info = \
            self.env.step(action)

        self.obs_cache        = deepcopy(obs)
        self.critic_obs_cache = deepcopy(critic_obs)
        self.need_hard_reset  = False

        #
        # NOTE: environments often return a reference to their info
        # dictionary. We sometimes add extra info to this, so we need
        # to make a copy to make sure things workout correctly (see
        # natural reward logic in filter wrappers for example).
        #
        return obs, critic_obs, reward, terminated, truncated, deepcopy(info)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, terminated, truncated,
                and info tuple.
        """
        return self._cache_step(action)

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        obs, critic_obs = self.env.reset()
        return obs, critic_obs

    def _env_soft_reset(self):
        """
            Check if our enviornment can perform a soft reset. If so, do it.
            If not, perform a standard reset.

            Returns:
                If our environment has a soft_reset method, we return the result
                of calling env.soft_reset. Otherwise, we return our obs_cache.
        """
        #
        # Tricky Business:
        # Perform a recursive call on all wrapped environments to check
        # for an ability to soft reset. Here's an example of how this works
        # in practice:
        #
        #   Imagine we wrap our environment like so:
        #   env = MultiAgentWrapper(ObservationNormalizer(ObservationClipper(env)))
        #
        #   When we call 'env.soft_reset()', we are calling
        #   ObservationClipper.soft_reset(). In this case, though, the clipper
        #   does not have direct access to the critic observations from the
        #   MultiAgentWrapper. So, we recursively travel back to the base
        #   wrapper, and return its soft_reset() results. Those results will
        #   then travel back up through the normalizer and clipper, being
        #   normalized and clipped, before being returned. It's basically
        #   identical to our "step" and "reset" calls, except each wrapper
        #   may have its own cache. The takeaway is that we ignore all caches
        #   other than the bottom level cache, which is transformed on its
        #   way out of the depths of recursion.
        #
        soft_reset = getattr(self.env, "soft_reset", None)

        if callable(soft_reset):
            return soft_reset()

        return self.obs_cache, self.critic_obs_cache

    def soft_reset(self):
        """
            Perform a "soft reset". This results in only performing the reset
            if the environment hasn't been reset since being created. This can
            allow us to start a new rollout from a previous rollout that ended
            near later in the environments timesteps.

            Returns:
                An observation.
        """
        if self.need_hard_reset or type(self.obs_cache) == type(None):
            return self.reset()

        return self._env_soft_reset()

    def render(self, **kw_args):
        """
            Render the environment.
        """
        return self.env.render(**kw_args)

    def save_info(self, path):
        """
            Save any info needed for loading the environment at a later time.

            Arguments:
                path    The path to save to.
        """
        self._check_env_save(path)

    def load_info(self, path):
        """
            Load any info needed for reinstating a saved environment.

            Arguments:
                path    The path to load from.
        """
        self._check_env_load(path)

    def augment_observation(self, obs):
        """
            If our environment has defined an observation augmentation
            method, we can access it here.

            Arguments:
                The observation to augment.

            Returns:
                The batch of augmented observations.
        """
        if self.can_augment_obs:
            return self.env.augment_observation(obs)
        else:
            raise NotImplementedError

    def augment_critic_observation(self, obs):
        """
            If our environment has defined a critic observation augmentation
            method, we can access it here.

            Arguments:
                The observation to augment.

            Returns:
                The batch of augmented observations.
        """
        if self.can_augment_obs:
            return self.env.augment_critic_observation(obs)
        else:
            raise NotImplementedError

    def _check_env_save(self, path):
        """
            Determine if our wrapped environment has a "save_info"
            method. If so, call it.

            Arguments:
                path    The path to save to.
        """
        save_info = getattr(self.env, "save_info", None)

        if callable(save_info):
            save_info(path)

    def _check_env_load(self, path):
        """
            Determine if our wrapped environment has a "load_info"
            method. If so, call it.

            Arguments:
                path    The path to load from.
        """
        load_info = getattr(self.env, "load_info", None)

        if callable(load_info):
            load_info(path)

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True or False.
        """
        batch_support = getattr(self.env, "supports_batched_environments", None)

        if callable(batch_support):
            return batch_support()
        else:
            return False

    def get_batch_size(self):
        """
            If any wrapped classes define this method, try to get the batch size
            from them. Otherwise, assume we have a single environment.

            Returns:
                Our batch size.
        """
        batch_size_getter = getattr(self.env, "get_batch_size", None)

        if callable(batch_size_getter):
            return batch_size_getter()
        else:
            return 1

    def has_wrapper(self, wrapper_class):
        """
            Recursively check to see if any of the wrappers around
            our environemnt match the given wrapper_class.

            Arguments:
                wrapper_class    The wrapper that we're interested in.

            Returns:
                True iff the environment is wrapped with wrapper_class.
        """
        if isinstance(self, wrapper_class):
            return True
        else:
            has_wrapper = getattr(self.env, "has_wrapper", None)

            if callable(has_wrapper):
                return has_wrapper(wrapper_class)

        return False

    def _finalize(self, status_dict):
        """
            Finalize the environment by recursing through all wrappers.

            Arguments:
                status_dict    The training status dict.
        """
        finalize = getattr(self.env, "finalize", None)

        if callable(finalize):
            finalize(status_dict)

    def finalize(self, status_dict):
        """
            Perform any finalizing acts.

            Arguments:
                status_dict    The training status dict.
        """
        self._finalize(status_dict)
        self.finalized = True

    def seed(self, seed):
        self.set_random_seed(seed)


class PPOEnvironmentWrapper(ABC):
    """
    The primary environment wrapper. ALL environments need to be
    wrapped in this.
    """

    def __init__(self,
                 env,
                 test_mode         = False,
                 add_agent_ids     = False,
                 agent_ids_as      = "float",
                 critic_view       = "policy",
                 policy_mapping_fn = None,
                 death_mask_reward = 0.0,
                 **kw_args):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        env: environment object
            The environment to wrap.
        test_mode: bool
            Are we in test mode?
        add_agent_ids: bool
            Should we add agent ids to the observations?
        agent_ids_as: str
            How to add the agent ids if add_agent_ids is True. Options
            are "float" and "one-hot".
        critic_view: str
            The view the critic should take. Options are "local",
            "policy", "global".
        policy_mapping_fn: function
            A function mapping agent ids to policy ids.
        death_mask_reward: float
            The reward to return for death-masked agents.
        """
        super(PPOEnvironmentWrapper, self).__init__(**kw_args)

        critic_view = critic_view.lower()
        assert critic_view in ["global", "local", "policy"]

        if critic_view == "policy" and policy_mapping_fn is None:
            msg  = "ERROR: policy_mapping_fn must be set when "
            msg += "critic_view is set to 'policy'."
            rank_print(msg)
            comm.Abort()

        self.env               = env
        self.all_done          = False
        self.null_actions      = {}
        self.add_agent_ids     = add_agent_ids
        self.agent_ids_as      = agent_ids_as
        self.critic_view       = critic_view
        self.policy_mapping_fn = policy_mapping_fn

        self._define_agent_ids()
        self.num_agents = len(self.agent_ids)

        if type(death_mask_reward) == dict:
            self.death_mask_reward = death_mask_reward

            for agent_id in self.agent_ids:
                assert agent_id in self.death_mask_reward

        elif isinstance(death_mask_reward, numbers.Number):
            self.death_mask_reward = {}

            for agent_id in self.agent_ids:
                self.death_mask_reward[agent_id] = death_mask_reward

        else:
            msg  = f"ERROR: unexpected type of {type(death_mask_reward)} "
            msg += "for death_mask_reward. Expecting type dict, int, or "
            msg += "float."
            rank_print(msg)
            comm.Abort()

        self.action_space             = Dict()
        self.observation_space        = Dict()
        self.critic_observation_space = Dict()

        self._define_multi_agent_spaces()
        self._define_critic_space()

        if callable(getattr(self.env, "augment_observation", None)):
            self.can_augment_obs = True

        self.agents_done = {a_id : False for a_id in self.agent_ids}

        id_options = ["float", "one-hot"]
        if self.agent_ids_as not in id_options:
            msg  = f"ERROR: agent_ids_as must be one of {id_options}, "
            msg += f"but recieved {self.agent_ids_as}."
            rank_print(msg)
            comm.Abort()

        self.agent_int_ids = {}
        for a_idx, a_id in enumerate(self.agent_ids):
            self.agent_int_ids[a_id] = a_idx

        self.agent_one_hot_ids = {}
        for a_idx, a_id in enumerate(self.agent_ids):
            one_hot = np.zeros(self.num_agents - 1)

            if a_idx > 0:
                one_hot[a_idx - 1] = 1

            self.agent_one_hot_ids[a_id] = one_hot.copy()

    def _expand_space_for_ids(self, space):
        """
            Expand a given space to include agent ids.

            NOTE: this currently only works for Box spaces.

            Argument:
                space    The space to expand.

            Returns:
                The space expanded for agent ids.
        """
        if issubclass(type(space), Box) or (
            old_gym_spaces is not None and issubclass(type(space), old_gym_spaces.Box)
        ):

            low   = space.low
            high  = space.high
            shape = space.shape

            low   = low.flatten()
            high  = high.flatten()

            if self.agent_ids_as == "float":
                shape = (reduce(lambda a, b: a * b, shape) + 1,)
                low   = np.concatenate((low, (0,)))

                #
                # NOTE: because we normalize ids, we set the cap
                # to 1.
                #
                high  = np.concatenate((high, (1,)))

            elif self.agent_ids_as == "one-hot":
                one_hot_size = self.num_agents - 1
                shape = (reduce(lambda a, b: a * b, shape) + one_hot_size,)
                low   = np.concatenate((low, np.zeros(one_hot_size)))
                high  = np.concatenate((high, np.ones(one_hot_size)))

            return Box(
                low   = low,
                high  = high,
                shape = shape,
                dtype = space.dtype)

        elif issubclass(type(space), Discrete):
            msg  = "ERROR: we do not support adding agent ids to "
            msg += "Discrete space observations."
            rank_print(msg)
            comm.Abort()
        else:
            msg  = "ERROR: we do not support adding agent ids to "
            msg += f"{type(space)} type space observations."
            rank_print(msg)
            comm.Abort()

    def _reset_done_agents(self):
        """
            Reset all agents to being not done.
        """
        for agent_id in self.agents_done:
            self.agents_done[agent_id] = False

    def _update_done_agents(self, done):
        """
            Update our dictionary of done agents.

            Arguments:
                done    The done dictionary from taking a step.
        """
        for agent_id in done:
            self.agents_done[agent_id] = done[agent_id]

    def _filter_done_agent_actions(self, actions):
        """
            Filter out the actions of agents that are done. This allows
            us to use death masking while preventing dead agents from
            sending masked actions to the simulation.

            Arguments:
                actions    The action dictionary to filter.
        """
        filtered_actions = {}

        for agent_id in actions:
            if not self.agents_done[agent_id]:
                filtered_actions[agent_id] = actions[agent_id]

        return filtered_actions

    def _get_done_dict(self, terminal, truncated):
        """
            Given terminal and truncated dictionaries, create a
            "done" dictionary.

            Arguments:
                terminal (dict)    The terminal dictionary.
                truncated (dict)   The truncated dictionary.

            Returns:
                A "done" dictionary.
        """
        done = {}
        for agent_id in terminal:
            assert agent_id in truncated
            done[agent_id] = terminal[agent_id] or truncated[agent_id]

        return done

    def _apply_death_mask(self, obs, reward, terminated, truncated, info):
        """
        Apply death masking. Note that this will alter the done
        agents to be not done.

        Parameters:
        -----------
        obs: dict
            The observation dictionary.
        reward: dict
            The reward dictionary.
        terminated: dict
            The terminated diciontary.
        truncated: dict
            The truncated diciontary.
        info: dict
            The info dictionary.

        Returns:
        --------
        tuple:
            A death masked tuple containing observations, rewards,
            terminations, truncations, and infos.
        """
        for agent_id in self.agent_ids:
            if self.agents_done[agent_id]:

                #
                # Case 1: an agent has died during the current step. In this
                # case, we want to keep the reward, observation, and info, but
                # we need to set done to False.
                #
                if agent_id in obs:
                    if not self.all_done:
                        terminated[agent_id] = False
                        truncated[agent_id]  = False

                #
                # Case 2: the agent died in a previous step. We now need to zero
                # out the observations and rewards and set info to empty.
                #
                else:
                    obs[agent_id] = np.zeros(
                        self.observation_space[agent_id].shape)
                    obs[agent_id] = obs[agent_id].astype(
                        self.observation_space[agent_id].dtype)

                    reward[agent_id] = self.death_mask_reward[agent_id]

                    #
                    # NOTE: all_done does not distinguish between terminated
                    # and truncated, but that doesn't matter here; if all
                    # agents are done (either way), we can offically say
                    # dead agents are terminated.
                    #
                    terminated[agent_id] = self.all_done
                    truncated[agent_id]  = False

                    info[agent_id] = {}

            elif agent_id not in obs:
                msg  = "ERROR: encountered an agent_id that is not done, but "
                msg += "it's missing from the observation. This may be a turn "
                msg += "based game, which is not yet supported."
                rank_print(msg)
                comm.Abort()

        return obs, reward, terminated, truncated, info

    def _add_agent_ids_to_obs(self, obs):
        """
            Add our agent ids to the given observations.

            Arguments:
                obs    The observation dictionary.

            Returns:
                The updated observation dictionary.
        """
        for a_id in obs:
            if self.agent_ids_as == "float":
                scaled_id = self.agent_int_ids[a_id] / self.num_agents
                obs[a_id] = np.concatenate(
                    (obs[a_id], (scaled_id,)))

            elif self.agent_ids_as == "one-hot":
                obs[a_id] = np.concatenate((obs[a_id],
                    self.agent_one_hot_ids[a_id])).astype(obs[a_id].dtype)

            else:
                msg  = "ERROR: encountered unknown value for "
                msg += f"agent_ids_as: {self.agent_ids_as}"
                rank_print(msg)
                comm.Abort()

        return obs

    def _define_critic_space(self):
        """
            Define the observation space for the critic. There are
            several options here.
        """
        #
        # Local view: the critics only see what the actor sees.
        #
        if self.critic_view == "local":
            for a_id in self.agent_ids:
                self.critic_observation_space[a_id] = \
                    self.observation_space[a_id]

        #
        # Global view: the critics see everything (observations from all
        # available agents).
        #
        elif self.critic_view == "global":
            for a_id in self.agent_ids:
                self.critic_observation_space[a_id] = \
                    gym.spaces.flatten_space(self.observation_space)

        #
        # Policy view: the critics see observations from all agents that
        # share policies.
        #
        elif self.critic_view == "policy":
            self.policy_spaces = {}

            #
            # First, map policy ids to shared spaces.
            #
            for a_id in self.agent_ids:
                policy_id = self.policy_mapping_fn(a_id)

                if policy_id not in self.policy_spaces:
                    self.policy_spaces[policy_id] = []

                self.policy_spaces[policy_id].append(
                    self.observation_space[a_id])

            #
            # Next, flatten the spaces so that each policy has a single space.
            #
            for policy_id in self.policy_spaces:
                self.policy_spaces[policy_id] = \
                    gym.spaces.flatten_space(Tuple(
                        self.policy_spaces[policy_id]))

            #
            # Lastly, we can assign spaces to agents.
            #
            for a_id in self.agent_ids:
                policy_id = self.policy_mapping_fn(a_id)
                self.critic_observation_space[a_id] = \
                    self.policy_spaces[policy_id]

        else:
            rank_print(f"ERROR: unknown critic_view, {self.critic_view}")
            comm.Abort()

    def _construct_global_critic_observation(self,
                                             obs,
                                             done):
        """
            Construct the global view observations for the
            critic. This view sees observations from all agents
            in the simulation.

            Arguments:
                obs    The agent observation dictionary.
                done   The done dictionary.

            Returns:
                The critic observations.
        """
        #
        # All agents will share the same critic observations.
        # Let's be memory sensitive here and only construct
        # one observation that all agents can share.
        #
        critic_obs = {}

        agent0 = next(iter(self.agent_ids))
        critic_shape = (1, reduce(lambda a, b: a * b,
            self.critic_observation_space[agent0].shape))

        critic_obs_data = np.zeros(critic_shape)

        critic_obs_data = critic_obs_data.astype(
            self.critic_observation_space[agent0].dtype)

        #
        # First pass: construct the shared observation.
        #
        start = 0
        for a_id in self.agent_ids:
            obs_size = reduce(lambda a, b: a * b,
                self.observation_space[a_id].shape)

            stop = start + obs_size

            #
            # There are two cases where we zero out the observations:
            #  1. The agent has died, but we're not all done (death masking).
            #  2. An agent hasn't acted in this turn ("turn masking"). We could
            #     also keep around previous turn observations in these cases,
            #     but it's unclear if that would be useful. TODO: investigate.
            #
            # NOTE: death masking is often handled by the "apply_death_mask"
            # method before this method is even called, so this logic here
            # is usually skipped. Dead agents will still have zero
            # observations.
            #
            if a_id not in obs or (done[a_id] and not self.all_done):
                mask = np.zeros(obs_size)
                mask = mask.astype(critic_obs_data.dtype)
                critic_obs_data[0, start : stop] = mask.flatten()

            elif a_id in obs:
                critic_obs_data[0, start : stop] = obs[a_id].flatten()

            start = stop

        #
        # Second pass: assign the shared observation.
        #
        for a_id in self.agent_ids:
            critic_obs[a_id] = critic_obs_data

        return critic_obs

    def _construct_policy_critic_observation(self,
                                             obs,
                                             done):
        """
            Construct the policy view observations for the
            critic. This view sees observations from all agents
            that share a policy.

            Arguments:
                obs    The agent observation dictionary.
                done   The done dictionary.

            Returns:
                The critic observations.
        """
        critic_obs  = {}
        policy_data = {}

        for policy_id in self.policy_spaces:
            data = np.zeros((1, reduce(lambda a, b : a * b,
                self.policy_spaces[policy_id].shape)))

            data = data.astype(
                self.policy_spaces[policy_id].dtype)

            policy_data[policy_id] = {"start" : 0, "data" : data}

        #
        # First pass: construct the shared observations.
        #
        for a_id in self.agent_ids:
            policy_id = self.policy_mapping_fn(a_id)

            obs_size = reduce(lambda a, b: a * b,
                self.observation_space[a_id].shape)

            start = policy_data[policy_id]["start"]
            stop  = start + obs_size

            #
            # There are two cases where we zero out the observations:
            #  1. The agent has died, but we're not all done (death masking).
            #  2. An agent hasn't acted in this turn ("turn masking"). We could
            #     also keep around previous turn observations in these cases,
            #     but it's unclear if that would be useful. TODO: investigate.
            #
            # NOTE: death masking is often handled by the "apply_death_mask"
            # method before this method is even called, so this logic here
            # is usually skipped. Dead agents will still have zero
            # observations.
            #
            if a_id not in obs or (done[a_id] and not self.all_done):
                mask      = np.zeros(obs_size)
                mask      = mask.astype(self.observation_space.dtype)
                agent_obs = mask.flatten()

            elif a_id in obs:
                agent_obs = obs[a_id].flatten()

            policy_data[policy_id]["data"][0][start : stop] = agent_obs

            policy_data[policy_id]["start"] = stop

        #
        # Second pass: assign the shared observations.
        #
        for a_id in self.agent_ids:
            policy_id        = self.policy_mapping_fn(a_id)
            critic_obs[a_id] = policy_data[policy_id]["data"]

        return critic_obs

    def _construct_critic_observation(self,
                                      obs,
                                      done):
        """
            Construct the critic observations.

            Arguments:
                obs    The agent observation dictionary.
                done   The agent done dictionary.

            Returns:
                The critic observations.
        """
        if self.critic_view == "global":
            return self._construct_global_critic_observation(obs, done)
        if self.critic_view == "local":
            return deepcopy(obs)
        if self.critic_view == "policy":
            return self._construct_policy_critic_observation(obs, done)
        else:
            rank_print(f"ERROR: unknown critic_view, {self.critic_view}.")
            comm.Abort()

    @abstractmethod
    def _define_agent_ids(self):
        """
            Abstract method for defining self.agent_ids.
        """
        return

    @abstractmethod
    def _define_multi_agent_spaces(self):
        """
            Abstract method for defining the action and observation
            spaces of our agents.
        """
        return

    def get_all_done(self):
        """
            Get the all_done boolean flag.
        """
        return self.all_done

    @abstractmethod
    def step(self, action):
        """
            Take a single step in the environment using the given
            action.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, terminated, truncated,
                and info tuple.
        """
        return

    @abstractmethod
    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation.
        """
        return

    def render(self, frame_pause = 0.0, **kw_args):
        """
            Render the environment.

            Arguments:
                frame_pause    Sleep frame_pause seconds before rendering.
        """
        if frame_pause > 0.0:
            time.sleep(frame_pause)

        return self.env.render(**kw_args)

    def seed(self, seed):
        """
            Set the seed for this environment.
        """
        self.env.seed(seed)


class VectorizedEnv(IdentityWrapper, Iterable):
    """
        Wrapper for "vectorizing" environments. As far as I can tell, this
        definition of vectorize is pretty specific to RL and refers to a way
        of reducing our environment episodes to "fixed-length trajecorty
        segments". This allows us to set our maximum timesteps per episode
        fairly low while still observing an episode from start to finish.
        This is accomplished by these "segments" that can (and often do) start
        in the middle of an episode.
        Vectorized environments are also often referred to as environment
        wrappers that contain multiple instances of environments, which then
        return batches of observations, rewards, etc. We currently support
        both.
    """

    def __init__(self,
                 env_generator,
                 num_envs = 1,
                 **kw_args):
        """
            Initialize our vectorized environment.

            Arguments:
                env_generator    A function that creates instances of our
                                 environment.
                num_envs         The number of environments to maintain.
        """
        super(VectorizedEnv, self).__init__(
            env_generator(),
            **kw_args)

        self.num_envs = num_envs
        self.envs     = np.array([None] * self.num_envs, dtype=object)
        self.iter_idx = 0
        self.steps    = np.zeros(self.num_envs, dtype=np.int32)

        if self.num_envs == 1:
            self.envs[0] = self.env
        else:
            for i in range(self.num_envs):
                #
                # Use a deepcopy in case users are passing in a reference.
                #
                self.envs[i] = deepcopy(env_generator())

    def set_random_seed(self, seed):
        """
            Set the random seed for the environment.

            Arguments:
                seed    The seed value.
        """
        for env_idx in range(self.num_envs):
            self.envs[env_idx].seed(seed)
            for agent_id in self.envs[env_idx].action_space:
                self.envs[env_idx].action_space[agent_id].seed(seed)

    def step(self, action):
        """
            Take a step in our environment with the given action.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state.

            Arguments:
                action    A dictionary mapping agent ids to actions.

            Returns:
                The resulting observation, critic_observation,
                reward, terminated, truncated, and info tuple,
                each being a dictionary.
        """
        #
        # If we're testing, we don't want to return a batch.
        #
        if self.test_mode:
            return self.single_step(action)

        return self.batch_step(action)


    def single_step(self, action):
        """
            Take a step in our environment with the given action.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state.

            Arguments:
                action    A dictionary mapping agent ids to actions.

            Returns:
                The resulting observation, critic_observation,
                reward, terminated, truncated, and info tuple,
                each being a dictionary.
        """
        obs, critic_obs, reward, terminated, truncated, info = \
            self.env.step(action)

        if self.env.get_all_done():
            for agent_id in info:
                info[agent_id]["terminal observation"] = deepcopy(obs[agent_id])

                info[agent_id]["terminal critic observation"] = \
                    deepcopy(critic_obs[agent_id])

            obs, critic_obs = self.env.reset()

        return obs, critic_obs, reward, terminated, truncated, info

    def batch_step(self, actions):
        """
            Take a step in our environment with the given actions.
            Since we're vectorized, we reset the environment when
            we've reached a "done" state, and we return a batch
            of step results. Since this is a batch step, we'll return
            a batch of results of size self.num_envs.

            Arguments:
                action    A dictionary mapping agent ids to batches
                          of actions.

            Returns:
                The resulting observation, critic_observation,
                reward, terminated, truncated, and info tuple,
                each being a dictionary.
        """
        batch_obs        = {}
        batch_critic_obs = {}
        batch_rewards    = {}
        batch_terminated = {}
        batch_truncated  = {}
        batch_infos      = {}

        #
        # Each agent keeps track of its own batches.
        #
        for agent_id in self.agent_ids:
            obs_shape = (self.num_envs,) + \
                self.observation_space[agent_id].shape

            critic_obs_shape = \
                (self.num_envs,) + \
                 self.critic_observation_space[agent_id].shape

            batch_obs[agent_id]        = np.zeros(obs_shape)
            batch_critic_obs[agent_id] = np.zeros(critic_obs_shape)
            batch_rewards[agent_id]    = np.zeros((self.num_envs, 1))
            batch_terminated[agent_id] = np.zeros((self.num_envs, 1)).astype(bool)
            batch_truncated[agent_id]  = np.zeros((self.num_envs, 1)).astype(bool)
            batch_infos[agent_id]      = np.array([None] * self.num_envs)

        env_actions = np.array([{}] * self.num_envs)
        for agent_id in actions:
            for b_idx in range(self.num_envs):
                env_actions[b_idx][agent_id] = actions[agent_id][b_idx]

        for env_idx in range(self.num_envs):
            act = env_actions[env_idx]

            obs, critic_obs, reward, terminated, truncated, info = \
                self.envs[env_idx].step(act)

            self.steps[env_idx] += 1

            if self.envs[env_idx].get_all_done():
                for agent_id in info:
                    info[agent_id]["terminal observation"] = \
                        deepcopy(obs[agent_id])

                    info[agent_id]["terminal critic observation"] = \
                        deepcopy(critic_obs[agent_id])

                obs, critic_obs = self.envs[env_idx].reset()

                self.steps[env_idx] = 0

            for agent_id in obs:
                batch_obs[agent_id][env_idx]        = obs[agent_id]
                batch_critic_obs[agent_id][env_idx] = critic_obs[agent_id]
                batch_rewards[agent_id][env_idx]    = reward[agent_id]
                batch_terminated[agent_id][env_idx] = terminated[agent_id]
                batch_truncated[agent_id][env_idx]  = truncated[agent_id]
                batch_infos[agent_id][env_idx]      = deepcopy(info[agent_id])

        self.obs_cache = deepcopy(batch_obs)
        self.critic_obs_cache = deepcopy(batch_critic_obs)
        self.need_hard_reset = False

        return (batch_obs, batch_critic_obs,
            batch_rewards, batch_terminated,
            batch_truncated, batch_infos)

    def reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation dictionaries for the actor
                and critic.
        """
        if self.test_mode:
            return self.single_reset()
        return self.batch_reset()

    def soft_reset(self):
        """
            Perform a soft reset if possible. A soft reset results
            in picking up where we last left off, rather than actually
            resetting the environment.

            Returns:
                The resulting observation dictionaries for the actor
                and critic.
        """
        if self.need_hard_reset:
            return self.reset()
        return self.obs_cache, self.critic_obs_cache

    def single_reset(self):
        """
            Reset the environment.

            Returns:
                The resulting observation and critic observation
                dictionaries.
        """
        obs, critic_obs = self.env.reset()
        return obs, critic_obs

    def batch_reset(self):
        """
            Reset the batch of environments.

            Returns:
                The resulting observation and critic observation
                dictionaries.
        """
        batch_obs        = {}
        batch_critic_obs = {}

        for env_idx in range(self.num_envs):
            obs, critic_obs = self.envs[env_idx].reset()

            self.steps[env_idx] = 0

            for agent_id in obs:
                if agent_id not in batch_obs:
                    obs_shape = (self.num_envs,) + \
                        self.observation_space[agent_id].shape
                    critic_obs_shape = \
                        (self.num_envs,) + \
                         self.critic_observation_space[agent_id].shape

                    batch_obs[agent_id] = np.zeros(obs_shape).astype(np.float32)
                    batch_critic_obs[agent_id] = \
                        np.zeros(critic_obs_shape).astype(np.float32)

                batch_obs[agent_id][env_idx]        = obs[agent_id]
                batch_critic_obs[agent_id][env_idx] = critic_obs[agent_id]

        return batch_obs, batch_critic_obs

    def __len__(self):
        """
            Represent our length as our batch size.

            Returns:
                The number of environments in our batch.
        """
        return self.num_envs

    def __next__(self):
        """
            Allow iteration through our array of environments.

            Returns:
                The next environment in our array.
        """
        if self.iter_idx < self.num_envs:
            env = self.envs[self.iter_idx]
            self.iter_idx += 1
            return env

        raise StopIteration

    def __iter__(self):
        """
            Allow iteration through our array of environments.

            Returns:
                Ourself as an iterable.
        """
        return self

    def __getitem__(self, idx):
        """
            Allow accessing environments by index.

            Arguments:
                idx    The index to the desired environment.

            Returns:
                The environment from self.envs located at index idx.
        """
        return self.envs[idx]

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True or False.
        """
        return True

    def get_batch_size(self):
        """
            Get the batch size.

            Returns:
                Return our batch size.
        """
        return len(self)
