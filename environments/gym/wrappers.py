"""
This module contains wrappers for gym environments.
"""
from ppo_and_friends.environments.ppo_env_wrappers import PPOEnvironmentWrapper
import numpy as np
from gymnasium.spaces import Box, Discrete
from ppo_and_friends.utils.mpi_utils import rank_print
from ppo_and_friends.utils.spaces import validate_observation_space
from abc import abstractmethod
from functools import reduce
import numbers

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()


class PPOGymWrapper(PPOEnvironmentWrapper):
    """
    OpenAI gym environments typically return numpy arrays
    for each step. This wrapper will convert these environments
    into a more multi-agent friendly setup, where each step/reset
    returns dictionaries mapping agent ids to their attributes.
    This will also return a critic observation along with the
    actor observation.
    """
    def __init__(self,
                 env,
                 *args,
                 **kw_args):

        if "critic_view" in kw_args:
            supported_views = ["local", "global"]

            if kw_args["critic_view"] not in supported_views:
                msg  = "ERROR: PPO-AF gym wrappers only support critic views "
                msg += f"{supported_views}, but received "
                msg += f"{kw_args['critic_view']}."
                rank_print(msg)
                comm.Abort()

        env = validate_observation_space(env)

        super(PPOGymWrapper, self).__init__(
            env,
            *args,
            **kw_args)

        self.random_seed = None

    def step(self, actions):
        """
        Take a step in the environment.

        Parameters:
        -----------
        actions: dict
            A dictionary mapping agent ids to actions.

        Returns:
        --------
        The observation, critic_observation, reward, done,
        and info tuple.
        """
        actions = self._filter_done_agent_actions(actions)

        obs, critic_obs, reward, terminated, truncated, info = \
            self._wrap_gym_step(
                *self._validate_step_return(
                    *self.env.step(
                        self._unwrap_action(actions))))

        return obs, critic_obs, reward, terminated, truncated, info

    def reset(self):
        """
        Reset the environment.

        Returns:
        --------
        The actor and critic observations.
        """
        obs, critic_obs = self._wrap_gym_reset(
            *self._validate_reset_return(
                *self.env.reset(seed = self.random_seed)))

        #
        # Gym versions >= 0.26 require the random seed to be set
        # when calling reset. Since we don't want the same exact
        # episode to reply every time we reset, we increment the
        # seed. This retains reproducibility while allow each episode
        # to vary.
        #
        if self.random_seed != None:
            self.random_seed += 1

        return obs, critic_obs

    @abstractmethod
    def _unwrap_action(self,
                       action):
        """
        An abstract method defining how to unwrap an action.

        Parameters:
        -----------
        action: dict
            A dictionary mapping agent ids to actions.

        Returns:
        --------
        Agent actions that the underlying environment can
        process.
        """
        return

    @abstractmethod
    def _wrap_gym_step(self,
                       obs,
                       reward,
                       terminated,
                       truncated,
                       info):
        """
        An abstract method defining how to wrap our enviornment
        step.

        Paramters:
        ----------
        obs: array-like or tuple
            The agent observations.
        reward: float or tuple
            The agent rewards.
        terminated: bool or tuple
            The agent termination flags.
        truncated: bool or tuple
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs, reward,
        terminated, truncated, info) s.t. each is a dictionary.
        """
        return

    @abstractmethod
    def _wrap_gym_reset(self,
                        obs,
                        info):
        """
        An abstract method defining how to wrap our enviornment
        reset.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        info: dict
            An info dictionary.

        Returns:
        --------
        A tuple of form (obs, critic_obs) s.t.
        each is a dictionary.
        """
        return

    def seed(self,
             seed):
        """
        Set the seed for this environment.

        Parameters:
        -----------
        seed: int
            The random seed.
        """
        if seed != None:
            assert type(seed) == int

        self.random_seed = seed

    @abstractmethod
    def _validate_step_return(self,
                              obs,
                              reward,
                              terminated,
                              truncated,
                              info):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: array-like or number
            The agent observations.
        reward: float
            The agent rewards.
        terminated: bool
            The agent termination flags.
        truncated: bool
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, reward,
        terminated, truncated, info).
        """
        return

    @abstractmethod
    def _validate_reset_return(self,
                               obs,
                               info):
        """
        Validate the return values from resetting our environment.

        Parameters:
        -----------
        obs: array-like or number
            The agent observations.
        info: dict
            An info dictionary.

        Returns:
        --------
        A tuple of form (obs, info).
        """
        return


class SingleAgentGymWrapper(PPOGymWrapper):
    """
    A wrapper for single agent gym environments.
    """

    def __init__(self,
                 env,
                 test_mode   = False,
                 **kw_args):
        """
        Parameters:
        ----------
        env: gym environment
            The gym environment to wrap.
        test_mode: bool
            Are we testing?
        """
        super(SingleAgentGymWrapper, self).__init__(
            env,
            test_mode,
            critic_view = "local",
            **kw_args)

        if self.add_agent_ids:
            msg  = "WARNING: adding agent ids is not applicable "
            msg += "for single agent simulators. Disregarding."
            rank_print(msg)

    def _define_agent_ids(self):
        """
        Define our agent_ids.
        """
        self.agent_ids  = ("agent0",)
        self.num_agents = 1

    def _define_multi_agent_spaces(self):
        """
        Define our multi-agent spaces. We have a single agent here,
        """
        for a_id in self.agent_ids:
            self.action_space[a_id]      = self.env.action_space
            self.observation_space[a_id] = self.env.observation_space

    def get_agent_id(self):
        """
        Get our only agent's id.

        Returns:
        --------
        Our agent's id.
        """
        if len(self.agent_ids) != 1:
            msg  = "ERROR: SingleAgentGymWrapper expects a single agnet, "
            msg += "but there are {}".format(len(self.agent_ids))
            rank_print(msg)
            comm.Abort()

        return self.agent_ids[0]

    def _unwrap_action(self,
                       action):
        """
        An method defining how to unwrap an action.

        Parameters:
        ----------
        action: dict
            A dictionary mapping agent ids to actions.

        Returns:
        --------
        A numpy array of actions.
        """
        agent_id   = self.get_agent_id()
        env_action = action[agent_id]

        if self.action_space[agent_id].shape == ():
            env_action = env_action.item()
        else:
            env_action = env_action.reshape(self.action_space[agent_id].shape)

        return env_action

    def _wrap_gym_step(self,
                       obs,
                       reward,
                       terminated,
                       truncated,
                       info):
        """
        A method defining how to wrap our enviornment
        step.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        reward: float
            The agent rewards.
        terminated: bool
            The agent termination flags.
        truncated: bool
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs, reward,
        terminated, truncated, info) s.t. each is a dictionary.
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space[agent_id].shape)

        if type(reward) == np.ndarray:
            reward = reward[0]

        reward = np.float32(reward)

        if terminated or truncated:
            self.all_done = True
        else:
            self.all_done = False

        obs        = {agent_id : obs}
        reward     = {agent_id : reward}
        truncated  = {agent_id : truncated}
        terminated = {agent_id : terminated}
        info       = {agent_id : info}

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        critic_obs = self._construct_critic_observation(obs, self.all_done)

        return obs, critic_obs, reward, terminated, truncated, info

    def _wrap_gym_reset(self,
                        obs,
                        info):
        """
        A method defining how to wrap our enviornment
        reset.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs) s.t.
        each is a dictionary.
        """
        agent_id = self.get_agent_id()

        #
        # HACK: some environments are buggy and don't follow their
        # own rules!
        #
        obs = obs.reshape(self.observation_space[agent_id].shape)
        obs = {agent_id : obs}

        done = {agent_id : False}
        self._update_done_agents(done)

        if self.add_agent_ids:
            obs = self._add_agent_ids_to_obs(obs)

        critic_obs = self._construct_critic_observation(obs, done)

        return obs, critic_obs

    def _validate_obs(self,
                      obs):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: array-like or number
            The agent observations.

        Returns:
        --------
        The agent observations.
        """
        #
        # Black magic: some environments (minigrid...) have dict observations.
        # We replace the observation space with SparseFlatteningDict, which
        # allows us to flatten and remove unsupported sub-spaces.
        #
        if isinstance(obs, dict):
            obs = self.env.observation_space.sparse_flatten_sample(obs)

        if isinstance(obs, numbers.Number):
            obs = np.array([obs])
        return obs

    def _validate_step_return(self,
                              obs,
                              reward,
                              terminated,
                              truncated,
                              info):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: array-like or number
            The agent observations.
        reward: float
            The agent rewards.
        terminated: bool
            The agent termination flags.
        truncated: bool
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, reward,
        terminated, truncated, info).
        """
        return self._validate_obs(obs), reward, terminated, truncated, info

    def _validate_reset_return(self,
                               obs,
                               info):
        """
        Validate the return values from resetting our environment.

        Parameters:
        -----------
        obs: array-like or number
            The agent observations.
        info: dict
            An info dictionary.

        Returns:
        --------
        A tuple of form (obs, info).
        """
        return self._validate_obs(obs), info


class SingleAgentGymSparseRewardWrapper(SingleAgentGymWrapper):

    def __init__(self,
                 env,
                 reward_trigger = 1.0,
                 reward_value   = 1.0,
                 sparse_value   = 0.0,
                 reward_freq    = 1,
                 **kw_args):
        """
        Parameters:
        ----------
        env: gym environment
            The gym environment to wrap.
        reward_trigger: float or tuple
            A reward is given only when this a reward of reward_trigger
            is given by the underlying environment. This can either be a
            float or a tuple of the inclusive [min, max] range.
        reward_value: float
            When reward_trigger is encountered reward_freq times, reward_value is
            the returned reward.
        sparse_value: float
            The value to return when the reward conditions have not been met.
        reward_freq: int
            reward_trigger must be encountered reward_freq times before
            reward_value is returned.
        """
        super(SingleAgentGymSparseRewardWrapper, self).__init__(
            env,
            **kw_args)

        if type(reward_trigger) == float:
            self.reward_trigger_min = reward_trigger
            self.reward_trigger_max = reward_trigger
        elif type(reward_trigger) == tuple:
            assert len(reward_trigger) == 2

            self.reward_trigger_min = reward_trigger[0]
            self.reward_trigger_max = reward_trigger[1]
        else:
            msg  = f"ERROR: reward_trigger must be a float or tuple "
            msg += f"but received {type(reward_trigger)}."
            rank_print(msg)
            comm.Abort()


        self.reward_value   = reward_value
        self.sparse_value   = sparse_value
        self.reward_freq    = reward_freq
        self.trigger_count  = 0

    def _wrap_gym_step(self, *args):
        """
        A method defining how to wrap our enviornment
        step.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        reward: float
            The agent rewards.
        terminated: bool
            The agent termination flags.
        truncated: bool
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs, reward,
        terminated, truncated, info) s.t. each is a dictionary.
        """
        agent_id = self.get_agent_id()

        obs, critic_obs, reward, terminated, truncated, info = super()._wrap_gym_step(*args)

        triggered = False
        if reward[agent_id] >= self.reward_trigger_min and reward[agent_id] <= self.reward_trigger_max:
            triggered = True
            self.trigger_count += 1

        reward[agent_id] = self.sparse_value

        if triggered and self.trigger_count % self.reward_freq == 0.0:
            reward[agent_id] = self.reward_value

        return obs, critic_obs, reward, terminated, truncated, info

    def _wrap_gym_reset(self, *args):
        """
        A method defining how to wrap our enviornment
        reset.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs) s.t.
        each is a dictionary.
        """
        self.trigger_count = 0
        return super()._wrap_gym_reset(*args)


class MultiAgentGymWrapper(PPOGymWrapper):
    """
    A wrapper for multi-agent gym environments.

    IMPORTANT: The following assumptions are made about the gym
    environment:

        1. All agent observations, actions, etc. are given in tuples
           s.t. each entry in the tuple corresponds to an agent.
        2. All agents must step at once. If an agent "dies", it still
           will return information every step.
    """

    def __init__(self,
                 env,
                 test_mode     = False,
                 add_agent_ids = True,
                 **kw_args):
        """
        Parameters:
        -----------
        env: gym enviornment
            The gym environment to wrap.
        test_mode: bool
            Are we in test mode?
        add_agent_ids: bool
            Should we add agent ids to the agent observations?
        """
        super(MultiAgentGymWrapper, self).__init__(
            env,
            test_mode,
            add_agent_ids = add_agent_ids,
            **kw_args)

    def _define_agent_ids(self):
        """
        Define our agent_ids.
        """
        self.num_agents = len(self.env.observation_space)
        self.agent_ids  = tuple(f"agent{i}" for i in range(self.num_agents))

    def _define_multi_agent_spaces(self):
        """
        Define our multi-agent spaces.
        """
        #
        # Some gym environments are buggy and require a reshape.
        #
        self.enforced_obs_shape = {}

        for a_idx, a_id in enumerate(self.agent_ids):
            if self.add_agent_ids:
                self.observation_space[a_id] = \
                    self._expand_space_for_ids(self.env.observation_space[a_idx])
            else:
                self.observation_space[a_id] = self.env.observation_space[a_idx]

            self.enforced_obs_shape[a_id] = \
                self.env.observation_space[a_idx].shape

            self.action_space[a_id] = self.env.action_space[a_idx]

    def _unwrap_action(self,
                       actions):
        """
        An method defining how to unwrap an action.

        Parameters:
        -----------
        actions: dict
            A dictionary mapping agent ids to actions.

        Returns:
        --------
        A tuple of actions.
        """
        gym_actions = np.array([None] * self.num_agents)

        for a_idx, a_id in enumerate(self.agent_ids):

            env_action = actions[a_id]

            if self.action_space[a_id].shape == ():
                gym_actions[a_idx] = env_action.item()
            else:
                gym_actions[a_idx] = env_action.reshape(self.action_space[a_id].shape)

        return tuple(gym_actions)

    def _wrap_gym_step(self,
                       obs,
                       reward,
                       terminated,
                       truncated,
                       info):
        """
        A method defining how to wrap our enviornment
        step.

        Parameters:
        -----------
        obs: tuple
            The agent observations.
        reward: tuple
            The agent rewards.
        terminated: tuple
            The agent termination flags.
        truncated: tuple
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs, reward,
        terminated, truncated, info) s.t. each is a dictionary.
        """
        wrapped_obs        = {}
        wrapped_reward     = {}
        wrapped_terminated = {}
        wrapped_truncated  = {}
        wrapped_info       = {}
        done_agents        = {}
        done_array         = np.zeros(len(self.agent_ids)).astype(bool)

        if truncated.any() and not truncated.all():
            msg  = "ERROR: truncation for one but not all agents in an "
            msg += "environment is not currently supported."
            rank_print(msg)
            comm.Abort()

        for a_idx, a_id in enumerate(self.agent_ids):
            agent_obs         = obs[a_idx]
            agent_reward      = reward[a_idx]
            agent_terminated  = terminated[a_idx]
            agent_truncated   = truncated[a_idx]
            agent_info        = info
            done_array[a_idx] = truncated[a_idx] or terminated[a_idx]

            #
            # HACK: some environments are buggy and don't follow their
            # own rules!
            #
            agent_obs = agent_obs.reshape(self.enforced_obs_shape[a_id])

            if type(agent_reward) == np.ndarray:
                agent_reward = agent_reward[0]

            agent_reward = np.float32(agent_reward)

            wrapped_obs[a_id]        = agent_obs
            wrapped_reward[a_id]     = agent_reward
            wrapped_info[a_id]       = agent_info
            wrapped_terminated[a_id] = agent_terminated
            wrapped_truncated[a_id]  = agent_truncated
            done_agents[a_id]        = agent_terminated or agent_truncated

        if done_array.all():
            self.all_done = True
        else:
            self.all_done = False

        if self.add_agent_ids:
            wrapped_obs = self._add_agent_ids_to_obs(wrapped_obs)

        self._update_done_agents(done_agents)

        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, wrapped_info = \
            self._apply_death_mask(
                wrapped_obs,
                wrapped_reward,
                wrapped_terminated,
                wrapped_truncated,
                wrapped_info)

        critic_obs  = self._construct_critic_observation(
            wrapped_obs, done_agents)

        return (wrapped_obs, critic_obs,
            wrapped_reward, wrapped_terminated,
            wrapped_truncated, wrapped_info)

    def _wrap_gym_reset(self,
                        obs,
                        info):
        """
        A method defining how to wrap our enviornment
        reset.

        Parameters:
        -----------
        obs: tuple
            The agent observations.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, critic_obs) s.t.
        each is a dictionary.
        """
        wrapped_obs  = {}
        wrapped_done = {}

        for a_idx, a_id in enumerate(self.agent_ids):
            agent_obs = obs[a_idx]

            #
            # HACK: some environments are buggy and don't follow their
            # own rules!
            #
            agent_obs = agent_obs.reshape(self.enforced_obs_shape[a_id])
            wrapped_obs[a_id] = agent_obs

            wrapped_done[a_id] = False

        self._update_done_agents(wrapped_done)

        if self.add_agent_ids:
            wrapped_obs = self._add_agent_ids_to_obs(wrapped_obs)

        critic_obs = self._construct_critic_observation(
            wrapped_obs, wrapped_done)

        return wrapped_obs, critic_obs

    def _validate_obs(self,
                      obs):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: array-like
            The agent observations.

        Returns:
        --------
        The agent observations.
        """
        for i in range(self.num_agents):

            #
            # Black magic: some environments (minigrid...) have dict observations.
            # We replace the observation space with SparseFlatteningDict, which
            # allows us to flatten and remove unsupported sub-spaces.
            #
            if isinstance(obs[i], dict):
                obs[i] = self.env.observation_space.sparse_flatten_sample(obs[i])

            if isinstance(obs[i], numbers.Number):
                obs[i] = np.array([obs[i]])
        return obs

    def _validate_step_return(self,
                              obs,
                              reward,
                              terminated,
                              truncated,
                              info):
        """
        Validate the return values from stepping in our environment.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        reward: float
            The agent rewards.
        terminated: bool
            The agent termination flags.
        truncated: bool
            The agent truncated flags.
        info: dict
            The agent info.

        Returns:
        --------
        A tuple of form (obs, reward,
        terminated, truncated, info).
        """
        return self._validate_obs(obs), reward, terminated, truncated, info

    def _validate_reset_return(self,
                               obs,
                               info):
        """
        Validate the return values from resetting our environment.

        Parameters:
        -----------
        obs: array-like
            The agent observations.
        info: dict
            An info dictionary.

        Returns:
        --------
        A tuple of form (obs, info).
        """
        return self._validate_obs(obs), info
