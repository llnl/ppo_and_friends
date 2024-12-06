"""
    Environment wrappers that send data through filtration of
    some kind. This can be things like normalization, clipping,
    augmenting, etc.
"""
from ppo_and_friends.utils.stats import RunningMeanStd
from ppo_and_friends.environments.ppo_env_wrappers import IdentityWrapper
from ppo_and_friends.utils.schedulers import CallableValue
import numpy as np
from copy import deepcopy
import pickle
import os
from ppo_and_friends.utils.mpi_utils import rank_print
from abc import ABC, abstractmethod
import sys

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

class ObservationFilter(IdentityWrapper, ABC):
    """
        An abstract class for filtering observations.
    """

    @abstractmethod
    def _filter_critic_observation(self, critic_obs):
        """
            Perform the local filtering on the critic observation.

            Arguments:
                critic_obs    The critic observation to filter.

            Returns:
                The filtered observation.
        """
        return

    @abstractmethod
    def _filter_local_observation(self, obs):
        """
            Perform the local filtering on the local observation.

            Arguments:
                obs    The local observation to filter.

            Returns:
                The filtered observation.
        """
        return

    def _apply_filters(self, local_obs, critic_obs):
        """
            Apply our filters.

            Arguments:
                local_obs      The individual actor observations.
                critic_obs     The observations for the critic.
        """
        local_obs  = self._filter_local_observation(local_obs)
        critic_obs = self._filter_critic_observation(critic_obs)
        return local_obs, critic_obs

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and filter the
            resulting observation.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, terminated, truncated,
                and info tuple.
        """
        obs, critic_obs, reward, terminated, truncated, info = \
            self.env.step(action)

        obs, critic_obs = self._apply_filters(obs, critic_obs)

        #
        # We need to cache the observation in case our lower level
        # wrappers don't have soft_reset defined.
        #
        self.obs_cache = deepcopy(obs)
        self.need_hard_reset = False

        return obs, critic_obs, reward, terminated, truncated, info

    def reset(self):
        """
            Reset the environment, and perform any needed filtering.

            Returns:
                The resulting observation.
        """
        obs, critic_obs = self.env.reset()
        return self._apply_filters(obs, critic_obs)

    def soft_reset(self):
        """
            Reset the environment, and perform any needed filtering.

            Returns:
                The resulting observation.
        """
        obs, critic_obs = self._env_soft_reset()
        return self._apply_filters(obs, critic_obs)


class ObservationNormalizer(ObservationFilter):
    """
        A wrapper for normalizing observations. This normalization
        method uses running statistics.
        NOTE: this uses implentations very similar to some of Stable
        Baslines' VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Should we update the statistics? We typically
                               have this enabled until testing.
                epsilon        A very small number to help avoid dividing by 0.
        """

        super(ObservationNormalizer, self).__init__(
            env,
            **kw_args)

        self.actor_running_stats = {}

        for agent_id in self.env.observation_space:
            self.actor_running_stats[agent_id] = RunningMeanStd(
                shape = self.env.observation_space[agent_id].shape)

        self.critic_running_stats = {}

        for agent_id in self.env.critic_observation_space:
            self.critic_running_stats[agent_id] = RunningMeanStd(
                shape = self.env.critic_observation_space[agent_id].shape)

        self.update_stats  = update_stats
        self.epsilon       = epsilon

    def _filter_local_observation(self, obs):
        """
            Send a local observation through the normalization process.

            Arguments:
                obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        if self.update_stats:
            for agent_id in obs:
                self.actor_running_stats[agent_id].update(obs[agent_id])

        return self.local_normalize(obs)

    def _filter_critic_observation(self, critic_obs):
        """
            Send a critic observation through the normalization process. This may
            called recursively by filter_critic_observation().

            Arguments:
                critic_obs    The critic observation to normalize.

            Returns:
                The normalized observation.
        """
        if self.update_stats:
            for agent_id in critic_obs:
                self.critic_running_stats[agent_id].update(critic_obs[agent_id])

        return self.critic_normalize(critic_obs)

    def local_normalize(self, obs):
        """
            A simple wrapper around self._local_normalize() that mitigates
            issues with memory references.

            Arguments:
                obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        for agent_id in obs:
            if type(obs[agent_id]) == np.ndarray:
                obs[agent_id] = self._local_normalize(
                    agent_id, deepcopy(obs[agent_id]))
            else:
                obs[agent_id] = self._local_normalize(
                    agent_id, obs[agent_id])

        return obs

    def _local_normalize(self, agent_id, agent_obs):
        """
            Normalize an observation using our running stats.

            Arguments:
                agent_id     The assocated agent id.
                agent_obs    The observation to normalize.

            Returns:
                The normalized observation.
        """
        agent_obs = (agent_obs - self.actor_running_stats[agent_id].mean) / \
            np.sqrt(self.actor_running_stats[agent_id].variance + self.epsilon)
        return agent_obs

    def critic_normalize(self, obs):
        """
            A simple wrapper around self._critic_normalize() that mitigates
            issues with memory references.

            Arguments:
                obs    The critic observation to normalize.

            Returns:
                The normalized observation.
        """
        for agent_id in obs:
            if type(obs[agent_id]) == np.ndarray:
                obs[agent_id] = self._critic_normalize(
                    agent_id, deepcopy(obs[agent_id]))
            else:
                obs[agent_id] = self._critic_normalize(
                    agent_id, obs[agent_id])

        return obs

    def _critic_normalize(self, agent_id, obs):
        """
            Normalize a critic observation using our running stats.

            Arguments:
                agent_id     The assocated agent id.
                obs          The critic observation to normalize.

            Returns:
                The normalized observation.
        """
        obs = (obs - self.critic_running_stats[agent_id].mean) / \
            np.sqrt(self.critic_running_stats[agent_id].variance + self.epsilon)
        return obs

    def _save_stats(self, path, file_name, stats):
        """
            Save out our running stats.

            Arguments: 
                path        The path to save to.
                file_name   The file name to save to.
                stats       The stats object to save.
        """
        out_file  = os.path.join(path, file_name)

        with open(out_file, "wb") as fh:
            pickle.dump(stats, fh)

    def _load_stats(self, path, file_name, backup_file_name, stats):
        """
            Load our running stats.

            Arguments: 
                path               The path to load from.
                file_name          The file name to load.
                backup_file_name   If file_name doesn't exist, rely on this
                                   file instead. This is useful when a processor
                                   file is missing, but we know rank 0 exists.
                stats              The stats object to load to.
        """
        in_file = os.path.join(path, file_name)

        if not os.path.exists(in_file):
            in_file = os.path.join(path, backup_file_name)

        with open(in_file, "rb") as fh:
            p_stats = pickle.load(fh)
            for key in p_stats:
                stats[key] = p_stats[key]

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        if self.test_mode:
            return

        file_name = "ActorRunningObsStats_{}.pickle".format(rank)
        self._save_stats(path, file_name, self.actor_running_stats)

        file_name = "CriticRunningObsStats_{}.pickle".format(rank)
        self._save_stats(path, file_name, self.critic_running_stats)

        self._check_env_save(path)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        if self.test_mode:
            actor_file_name  = "ActorRunningObsStats_0.pickle"
            critic_file_name = "CriticRunningObsStats_0.pickle"
        else:
            actor_file_name  = "ActorRunningObsStats_{}.pickle".format(rank)
            critic_file_name = "CriticRunningObsStats_{}.pickle".format(rank)

        backup_actor_file_name  = "ActorRunningObsStats_0.pickle"
        backup_critic_file_name = "CriticRunningObsStats_0.pickle"

        self._load_stats(path, actor_file_name,
            backup_actor_file_name, self.actor_running_stats)

        self._load_stats(path, critic_file_name,
            backup_critic_file_name, self.critic_running_stats)

        self._check_env_load(path)


class RewardNormalizer(IdentityWrapper):
    """
        This wrapper uses running statistics to normalize rewards.
        NOTE: some of this logic comes from Stable Baseline's
        VecNormalize.
    """

    def __init__(self,
                 env,
                 update_stats = True,
                 epsilon      = 1e-8,
                 gamma        = 0.99,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env            The environment to wrap.
                update_stats   Whether or not to update our running stats. This
                               is typically set to true until testing.
                epsilon        A very small value to help avoid 0 divisions.
                gamma          A discount factor for our running stats.
        """
        super(RewardNormalizer, self).__init__(
            env,
            **kw_args)

        self.running_stats = {}
        for agent_id in self.env.agent_ids:
            self.running_stats[agent_id] = RunningMeanStd(shape=())

        self.update_stats  = update_stats
        self.epsilon       = epsilon
        self.gamma         = gamma

        #
        # We might be wrapping an environment that supports batches.
        # if so, we need to be able to correlate a running reward with
        # each environment instance.
        #
        self.batch_size = self.get_batch_size()

        self.running_reward = {}
        for agent_id in self.env.agent_ids:
            self.running_reward[agent_id] = np.zeros(self.batch_size)

    def step(self, action):
        """
            Take a single step in the environment using the given
            action, update the running stats, and normalize the
            resulting reward.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation, reward, terminated, truncated,
                and info tuple.
        """
        obs, critic_obs, reward, terminated, truncated, info = \
            self._cache_step(action)

        agent_dones = {}
        for agent_id in self.env.agent_ids:
            agent_dones[agent_id] = np.logical_or(terminated[agent_id],
                truncated[agent_id])

        for agent_id in reward:

            if self.update_stats:
                for env_idx in range(self.batch_size):
                    self.running_reward[agent_id][env_idx] = \
                        self.running_reward[agent_id][env_idx] * \
                        self.gamma + reward[agent_id][env_idx]

                    self.running_stats[agent_id].update(
                        self.running_reward[agent_id])

            if self.batch_size > 1:
                where_done = np.where(agent_dones[agent_id])[0]
                self.running_reward[agent_id][where_done] = 0.0

            elif agent_dones[agent_id]:
                self.running_reward[agent_id][0] = 0.0

            #
            # NOTE: when training, we always receive an ndarray. When testing,
            # it's a flat value.
            #
            if type(reward[agent_id]) == np.ndarray:
                batch_size = reward[agent_id].shape[0]

                for b_idx in range(batch_size):
                    if "natural reward" not in info[agent_id][b_idx]:
                        info[agent_id][b_idx]["natural reward"] = \
                            deepcopy(reward[agent_id][b_idx])
            else:
                if "natural reward" not in info[agent_id]:
                    info[agent_id]["natural reward"] = \
                        deepcopy(reward[agent_id])

            reward[agent_id] = self.normalize(agent_id, reward[agent_id])

        return obs, critic_obs, reward, terminated, truncated, info

    def normalize(self, agent_id, agent_reward):
        """
            A simple wrapper around self._normalize() that mitigates
            issues with memory references.

            Arguments:
                agent_id        The associated agent id.
                agent_reward    The reward to normalize.

            Returns:
                The normalized reward.
        """
        if type(agent_reward) == np.ndarray:
            return self._normalize(agent_id, deepcopy(agent_reward))
        return self._normalize(agent_id, agent_reward)

    def _normalize(self, agent_id, agent_reward):
        """
            Normalize our reward using Stable Baseline's approach.

            Arguments:
                agent_id        The associated agent_id.
                agent_reward    The reward to normalize.

            Returns:
                The normalized reward.
        """
        agent_reward /= np.sqrt(
            self.running_stats[agent_id].variance + self.epsilon)
        return agent_reward

    def save_info(self, path):
        """
            Save out our running stats, and check if our wrapped
            environment needs to perform any more info saves.

            Arguments:
                path    The path to save to.
        """
        if self.test_mode:
            return

        file_name = "RunningRewardsStats_{}.pickle".format(rank)
        out_file  = os.path.join(path, file_name)

        with open(out_file, "wb") as fh:
            pickle.dump(self.running_stats, fh)

        self._check_env_save(path)

    def load_info(self, path):
        """
            Load our running stats and check to see if our wrapped
            environment needs to load anything.

            Arguments:
                path    The path to load from.
        """
        if self.test_mode:
            file_name = "RunningRewardsStats_0.pickle"
        else:
            file_name = "RunningRewardsStats_{}.pickle".format(rank)

        in_file = os.path.join(path, file_name)

        if not os.path.exists(in_file):
            file_name = "RunningRewardsStats_0.pickle"
            in_file   = os.path.join(path, file_name)

        with open(in_file, "rb") as fh:
            p_stats = pickle.load(fh)
            for key in p_stats:
                self.running_stats[key] = p_stats[key]

        self._check_env_load(path)


class GenericClipper(IdentityWrapper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                clip_range  The range to clip our rewards into. This can be
                            either a real number or a class that inherits from
                            utils/StatusScheduler.
        """

        super(GenericClipper, self).__init__(
            env,
            **kw_args)

        min_callable = None
        max_callable = None

        if callable(clip_range[0]):
            min_callable = clip_range[0]
        else:
            min_callable = CallableValue(clip_range[0])

        if callable(clip_range[1]):
            max_callable = clip_range[1]
        else:
            max_callable = CallableValue(clip_range[1])

        self.clip_range = (min_callable, max_callable)

    def finalize(self, status_dict):
        """
            Finalize our clip range, which might require a status dict.

            Arguments:
                status_dict    The training status dict.
        """
        self.clip_range[0].finalize(status_dict)
        self.clip_range[1].finalize(status_dict)
        self.finalized = True

        self._finalize(status_dict)

    def get_clip_range(self):
        """
            Get the current clip range.

            Returns:
                A tuple containing the clip range as (min, max).
        """
        min_value = self.clip_range[0]()
        max_value = self.clip_range[1]()

        return (min_value, max_value)

    def _apply_agent_clipping(self, agent_dict):
        """
            Apply clipping to all agent values.

            Arguments:
                agent_dict    A dictionary mapping agent ids to the values
                              needing to be clipped.

            Returns:
                A new agent dictionary s.t. all values have been clipped.
        """
        clipped_dict = {}
        for agent_id in agent_dict:
            clipped_dict[agent_id] = self._clip(agent_dict[agent_id])
        return clipped_dict

    def _clip(self, val):
        """
            Clip a value to our clip range.

            Arguments:
                val    The value to clip.

            Returns:
                The clipped value.
        """
        min_value, max_value = self.get_clip_range()
        return np.clip(val, min_value, max_value)


class ObservationClipper(ObservationFilter, GenericClipper):
    """
        An environment wrapper that clips observations.
    """

    def __init__(self,
                 env,
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                clip_range  The range to clip our observations into.
        """
        super(ObservationClipper, self).__init__(
            env,
            clip_range  = clip_range,
            **kw_args)

    def _filter_critic_observation(self, obs):
        """
            A simple wrapper for clipping the critic observation.

            Arguments:
                obs    The observation to clip.

            Returns:
                The clipped observation.
        """
        return self._apply_agent_clipping(obs)

    def _filter_local_observation(self, obs):
        """
            A simple wrapper for clipping local the observation.

            Arguments:
                obs    The observation to clip.

            Returns:
                The clipped observation.
        """
        return self._apply_agent_clipping(obs)


class RewardClipper(GenericClipper):
    """
        A wrapper for clipping rewards.
    """

    def __init__(self,
                 env,
                 clip_range  = (-10., 10.),
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env         The environment to wrap.
                clip_range  The range to clip our rewards into.
        """

        super(RewardClipper, self).__init__(
            env,
            clip_range  = clip_range,
            **kw_args)

    def step(self, actions):
        """
            Take a single step in the environment using the given
            actions, and clip the resulting reward.

            Arguments:
                actions    A dictionary mapping agent ids to their actions.

            Returns:
                The resulting observation, critic_observation, reward,
                terminated, truncated, and info tuple.
        """
        obs, critic_obs, reward, terminated, truncated, info = \
            self._cache_step(actions)

        for agent_id in reward:
            #
            # NOTE: when training, we always receive an ndarray. When testing,
            # it's a flat value.
            #
            if type(reward[agent_id]) == np.ndarray:
                batch_size = reward[agent_id].shape[0]

                for b_idx in range(batch_size):
                    if "natural reward" not in info[agent_id][b_idx]:
                        info[agent_id][b_idx]["natural reward"] = \
                            deepcopy(reward[agent_id][b_idx])
            else:
                if "natural reward" not in info[agent_id]:
                    info[agent_id]["natural reward"] = \
                        deepcopy(reward[agent_id])

        reward = self._apply_agent_clipping(reward)

        return obs, critic_obs, reward, terminated, truncated, info


# FIXME: this needs to be tested after MA refactor.
class ObservationAugmentingWrapper(IdentityWrapper):
    """
        This wrapper expects the environment to have a method named
        'augment_observation' that can be utilized to augment an observation
        to create a batch of obserations. Each instance of observation in the
        batch will be coupled with identical terminated, truncated, and reward.
        This is to help a policy learn that a particular augmentation does
        not affect the learned policy.
    """

    def __init__(self,
                 env,
                 **kw_args):
        """
            Initialize the wrapper.

            Arguments:
                env   The environment to wrap.
        """

        super(ObservationAugmentingWrapper, self).__init__(
            env,
            **kw_args)

        self.test_idx = -1
        aug_func      = getattr(env, "augment_observation", None)

        if type(aug_func) == type(None):
            msg  = "ERROR: env must define 'augment_observation' in order "
            msg += "to qualify for the ObservationAugmentingWrapper class."
            rank_print(msg)
            comm.Abort()

        self.batch_size = self.aug_reset().shape[0]

    def step(self, action):
        """
            Take a single step in the environment using the given
            action. If we're in test mode, we don't augment. Otherwise,
            call the aug_step method.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                The resulting observation(s), reward(s),
                terminated(s), truncated(s), and info(s).
        """
        #TODO: update to handle soft resets.
        if self.test_mode:
            return self.aug_test_step(action)
        return self.aug_step(action)

    def aug_step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation, and set up the return values as a batch.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                A batch of observations, rewards, terminated, and
                truncated  along with a single info dictionary. The observations
                will contain augmentations of the original observation.
        """
        # TODO: update for soft resets.
        obs, critic_obs, reward, terminated, truncated, info = \
            self.env.step(action)

        batch_obs        = self.env.augment_observation(obs)
        batch_critic_obs = self.env.augment_critic_observation(critic_obs)
        batch_size       = batch_obs.shape[0]

        batch_rewards    = {}
        batch_terminated = {}
        batch_truncated  = {}
        batch_infos      = {}

        # TODO: update for terminal critic obs
        for agent_id in obs:
            if "terminal observation" in info[0]:
                batch_infos[agent_id] = np.array([None] * batch_size,
                    dtype=object)

                terminal_obs = info[agent_id][0]["terminal observation"]
                terminal_obs = self.env.augment_observation(terminal_obs)

                for i in range(batch_size):
                    i_info = deepcopy(info[agent_id][0])
                    i_info["terminal observation"] = deepcopy(terminal_obs[i])
                    batch_infos[agent_id][i] = i_info
            else:
                batch_infos[agent_id] = np.tile((info[agent_id][0],), batch_size)

            batch_rewards[agent_id] = np.tile((reward[agent_id],), batch_size)
            batch_terminated[agent_id] = \
                np.tile((terminated[agent_id],), batch_size).astype(bool)
            batch_truncated[agent_id] = \
                np.tile((truncated[agent_id],), batch_size).astype(bool)

            batch_rewards[agent_id] = \
                batch_rewards[agent_id].reshape((batch_size, 1))
            batch_terminated[agent_id] = \
                batch_terminated[agent_id].reshape((batch_size, 1))
            batch_truncated[agent_id] = \
                batch_truncated[agent_id].reshape((batch_size, 1))
            batch_infos[agent_id] = batch_infos[agent_id].reshape((batch_size))

        return (batch_obs, batch_critic_obs, batch_rewards,
            batch_terminated, batch_truncated, batch_infos)

    def aug_test_step(self, action):
        """
            Take a single step in the environment using the given
            action, allow the environment to augment the returned
            observation. Since we're in test mode, we return a single
            instance from the batch.

            NOTE: the action is expected to be a SINGLE action. This does
            not currently support multiple environment instances.

            Arguments:
                action    The action to take.

            Returns:
                Observation, reward, terminated, truncated, and
                info (possibly augmented).
        """
        obs, critic_obs, reward, terminated, truncated, info = \
            self.env.step(action)

        batch_obs        = self.env.augment_observation(obs)
        batch_critic_obs = self.env.augment_critic_observation(critic_obs)
        batch_size       = batch_obs.shape[0]

        if self.test_idx < 0:
            self.test_idx = np.random.randint(0, batch_size)

        for agent_id in batch_obs:
            batch_obs[agent_id] = batch_obs[agent_id][self.test_idx]
            batch_critic_obs[agent_id] = \
                batch_critic_obs[agent_id][self.test_idx]

        # TODO: update for terminal critic obs
        for agent_id in info:
            if "terminal observation" in info[agent_id]:
                terminal_obs = info[agent_id]["terminal observation"]
                terminal_obs = self.env.augment_observation(terminal_obs)
                info[agent_id]["terminal observation"] = \
                    deepcopy(terminal_obs[self.test_idx])

        return batch_obs, batch_critic_obs, reward, terminated, truncated, info

    def reset(self):
        """
            Reset the environment. If we're in test mode, we don't augment the
            resulting observations. Otherwise, augment them before returning.

            Returns:
                The resulting observation(s).
        """
        if self.test_mode:
            return self.aug_test_reset()
        return self.aug_reset()

    def aug_reset(self):
        """
            Reset the environment, and return a batch of augmented observations.

            Returns:
                The resulting observations.
        """
        obs, critic_obs = self.env.reset()

        obs = self.env.augment_observation(obs)
        critic_obs = self.env.augment_critic_observation(critic_obs)

        return obs, critic_obs

    def aug_test_reset(self):
        """
            Reset the environment, and return a single observations which may
            or may not be augmented.

            Returns:
                The resulting observation (possibly augmented).
        """
        obs, critic_obs = self.env.reset()

        obs = self.env.augment_observation(obs)
        critic_obs = self.env.augment_critic_observation(critic_obs)
        batch_size = obs.shape[0]

        if self.test_idx < 0:
            self.test_idx = np.random.randint(0, batch_size)

        for agent_id in obs:
            obs[agent_id] = obs[agent_id][self.test_idx]
            critic_obs[agent_id] = critic_obs[agent_id][self.test_idx]

        return obs, critic_obs

    def get_batch_size(self):
        """
            If any wrapped classes define this method, try to get the batch size
            from them. Otherwise, assume we have a single environment.

            Returns:
                Return our batch size.
        """
        return self.batch_size

    def supports_batched_environments(self):
        """
            Determine whether or not our wrapped environment supports
            batched environments.

            Return:
                True.
        """
        return True
