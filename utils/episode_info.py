import numpy as np
import gc
from torch.utils.data import Dataset
import sys
import torch
from ppo_and_friends.utils.mpi_utils import rank_print
from abc import ABC, abstractmethod

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

def combine_episode_advantages(episodes,
                               list_combine_func = list.extend):
    """
    Combine the advantages across several episodes into one.

    Parameters:
    -----------
    episodes: array-like
        An array of PPOEpisode objects.
    list_combine_func: function
        The function to use for combining lists from each episode.

    Returns:
    --------
    list:
        The combined advantages.
    """
    advantages = []

    for ep in episodes:
        if not ep.is_finished:
            msg  = "ERROR: attempting to build a batch using "
            msg += "an incomplete episode! Bailing..."
            rank_print(msg)
            comm.Abort()

        list_combine_func(advantages, ep.advantages)

    return advantages

def combine_episodes(episodes,
                     build_hidden_states,
                     list_combine_func = list.extend,
                     release_data      = True):
    """
    Combine a series of episode info objects. This function iterates
    through an array of episodes and concatenates all of their datasets
    together.

    Parameters:
    -----------
    episodes: array-like
        An array of PPOEpisode objects.
    build_hidden_states: bool
        Whether or not to combine hidden states.
    list_combine_func: function
        The function to use for combining lists from each episode.
    release_data: bool
        Release data from episodes after they've been merged?

    Returns:
    --------
    tuple:
        A tuple containing the combined data from all episodes.
    """
    actions              = []
    raw_actions          = []
    critic_observations  = []
    observations         = []
    next_observations    = []
    rewards_to_go        = []
    log_probs            = []
    ep_lens              = []
    advantages           = []
    values               = []
    actor_hidden         = []
    critic_hidden        = []
    actor_cell           = []
    critic_cell          = []
    total_timestates     = 0

    for ep in episodes:

        if not ep.is_finished:
            msg  = "ERROR: attempting to build a batch using "
            msg += "an incomplete episode! Bailing..."
            rank_print(msg)
            comm.Abort()

        total_timestates += ep.length

        list_combine_func(actions, ep.actions)
        list_combine_func(raw_actions, ep.raw_actions)
        list_combine_func(observations, ep.observations)
        list_combine_func(next_observations, ep.next_observations)
        list_combine_func(rewards_to_go, ep.rewards_to_go)
        list_combine_func(log_probs, ep.log_probs)
        list_combine_func(advantages, ep.advantages)
        list_combine_func(values, ep.values)
        list_combine_func(critic_observations, ep.critic_observations)

        #
        # We always append length because they are integers.
        #
        ep_lens.append(ep.length)

        if build_hidden_states:
            list_combine_func(actor_hidden, ep.actor_hidden)
            list_combine_func(critic_hidden, ep.critic_hidden)

            list_combine_func(actor_cell, ep.actor_cell)
            list_combine_func(critic_cell, ep.critic_cell)

        if release_data:
            ep.release_data()

    return (
        actions,
        raw_actions,
        critic_observations,
        observations,
        next_observations,
        rewards_to_go,
        log_probs,
        ep_lens,
        advantages,
        values,
        actor_hidden,
        critic_hidden,
        actor_cell,
        critic_cell,
        total_timestates)


class PPOEpisode(ABC):

    def __init__(self, *args, **kw_args):
        """
        An abstract container for tracking episodes from PPO rollouts.
        """
        self.is_finished         = False
        self.actions             = None
        self.raw_actions         = None
        self.critic_observations = None
        self.observations        = None
        self.next_observations   = None
        self.rewards_to_go       = None
        self.log_probs           = None
        self.advantages          = None
        self.values              = None
        self.actor_hidden        = None
        self.critic_hidden       = None
        self.actor_cell          = None
        self.critic_cell         = None
        self.total_timestates    = None
        self.length              = 0

    @abstractmethod
    def compute_advantages(self, *args, **kw_args):
        """
        Compute the advantages of our agents.
        """
        raise NotImplementedError


class EpisodeInfo(PPOEpisode):

    def __init__(self,
                 starting_ts    = 0,
                 use_gae        = False,
                 gamma          = 0.99,
                 lambd          = 0.95,
                 bootstrap_clip = (-10., 10.)):
        """
        A container for storing episode information.

        Parameters:
        -----------
        starting_ts: int
            The timestep that this episode starts at.
        use_gae: bool
            Should we use the Generalized Advantage
            Estimation algorithm when calculating advantages?
        gamma: float
            The discount factor to apply when calculating
            discounted sums. This is used for advantages and
            "rewards-to-go" (expected discounted returns).
        labmd: float
            A "smoothing" factor used in GAE.
        bootstrap_clip: tuple or None
            A value to clip our bootstrapped rewards to if enabled. Otherwise,
            None.
        """
        super().__init__()

        self.starting_ts              = starting_ts
        self.ending_ts                = -1
        self.use_gae                  = use_gae
        self.gamma                    = gamma
        self.lambd                    = lambd
        self.bootstrap_clip           = bootstrap_clip
        self.critic_observations      = []
        self.observations             = []
        self.next_observations        = []
        self.actions                  = []
        self.raw_actions              = []
        self.log_probs                = []
        self.rewards                  = []
        self.values                   = []
        self.actor_hidden             = []
        self.critic_hidden            = []
        self.actor_cell               = []
        self.critic_cell              = []
        self.rewards_to_go            = None
        self.advantages               = None
        self.length                   = 0
        self.is_finished              = False
        self.has_hidden_states        = False

    def compute_discounted_sums(self, array, gamma):
        """
        Compute the discounted sums from a given array,
        which is assumed to be in temmporal order,
        [t0, t1, t2, ..., tn], where t0 is a 'value' at
        time 0, and tn is a 'value' at time n. Note that value
        here is not to be confused with the value from a value
        function; it's just some number. It could be a reward,
        a value from a value function, or whatever else you'd like.

        The discounted value at time t, DVt, follows the recursive formula
            DVt = DVt + (gamma * DVt+1)
        Such that all future values are considered in the current DV but
        at a discount.

        That is,
            DSn     = tn
            DS(n-1) = t(n-1) + (gamma * tn)
            ...
            DS0     = t0 + (gamma * t1) + (gamma^2 * t2) + ...
                      + (gamma^n + tn)

        Parameters:
        -----------
        array: np.ndarray
            The array to calculate a discounted sum for.

        Returns:
        --------
        A numpy array containing the discounted sums.
        """
        cumulative_array = np.zeros(len(array))
        last_idx         = len(array) - 1
        d_sum = 0

        for idx, val in enumerate(array[::-1]):
            d_sum = val + gamma * d_sum
            cumulative_array[last_idx - idx] = d_sum

        return cumulative_array

    def _compute_gae_advantages(self,
                                padded_values,
                                rewards):
        """
        Compute the General Advantage Estimates. This follows the
        general GAE equation.

        Parameters:
        -----------
        padded_values:np.ndarray
            An array of values from this epsiode with one
            extra value added to the end. This will either
            be a 0 (if the episode finished) or a repeat
            of the last value.

        Returns:
        --------
        An array containing the GAEs.
        """
        if np.isinf(padded_values).any():
            msg  = "ERROR: inf encountered in padded values while "
            msg += "computing gae advantages!"
            rank_print(msg, debug = True)
            comm.Abort()

        deltas = rewards + (self.gamma * padded_values[1:]) - \
            padded_values[:-1]

        sum_gamma = self.gamma * self.lambd
        return self.compute_discounted_sums(deltas.tolist(), sum_gamma)

    def _compute_standard_advantages(self):
        """
        Use a standard method for computing advantages.
        Typically, we use Q - values.
        """
        advantages = self.rewards_to_go - self.values
        return advantages

    def add_info(self,
                 observation,
                 next_observation,
                 raw_action,
                 action,
                 value,
                 log_prob,
                 reward,
                 critic_observation = np.empty(0),
                 actor_hidden       = np.empty(0),
                 actor_cell         = np.empty(0),
                 critic_hidden      = np.empty(0),
                 critic_cell        = np.empty(0)):
        """
        Add info from a single step in an episode. These should be
        added consecutively, in the order they are encountered.

        Parameters:
        -----------
        observation: np.ndarray
            The observation eliciting our action.
        next_observati: np.ndarray
            The observation resulting from our action.
        raw_action: np.ndarray
            The un-altered action (there are times
            when we squash our actions into a new
            range. This value is pre-squash).
        action: np.ndarray or float
            The action taken at this step.
        value: float
            The predicted value at this step (from the critic).
        log_prob: float
            The log probability calculated at this step.
        reward: float
            The reward received at this step.
        critic_observation: np.ndarray
            The critic observation used in multi-
            agent environments eliciting our action.
        actor_hidden: np.ndarray
            The hidden state of the actor iff the
            actor is an lstm.
        actor_cell: np.ndarray
            The cell state of the actor iff the
            actor is an lstm.
        critic_hidden: np.ndarray
            The hidden state of the critic iff the
            critic is an lstm.
        critic_cell: np.ndarray
            The cell state of the critic iff the
            critic is an lstm.
        """

        if type(raw_action) == np.ndarray and len(raw_action.shape) > 1:
            raw_action = raw_action.squeeze()

        if type(action) == np.ndarray and len(action.shape) > 1:
            action = action.squeeze()

        self.observations.append(observation)
        self.next_observations.append(next_observation)
        self.actions.append(action)
        self.raw_actions.append(raw_action)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.critic_observations.append(critic_observation)

        ac_hidden_check = np.array(
            (len(actor_hidden),
             len(actor_cell),
             len(critic_hidden),
             len(critic_cell))).astype(np.int32)

        if ( (ac_hidden_check > 0).any() and
             (ac_hidden_check == 0).any() ):

            msg  = "ERROR: if hidden state is provided for either the actor "
            msg += "or the critic, both most be provided, but we received "
            msg += "only one. Bailing..."
            rank_print(msg)
            comm.Abort()

        if len(actor_hidden) > 0:

            self.has_hidden_states = True

            self.actor_hidden.append(
                actor_hidden.detach().cpu().numpy())

            self.critic_hidden.append(
                critic_hidden.detach().cpu().numpy())

            self.actor_cell.append(
                actor_cell.detach().cpu().numpy())

            self.critic_cell.append(
                critic_cell.detach().cpu().numpy())

    def compute_advantages(self):
        """
        Compute our advantages using either a standard formula or GAE.
        """
        #
        # TODO: we should have an option to pass in a container to
        # put advantages into. This would save on some space.
        #
        if self.use_gae:
            padded_values = np.concatenate((self.values, (self.ending_value,)))
            padded_values = padded_values.astype(np.float32)

            self.advantages = self._compute_gae_advantages(
                padded_values,
                self.rewards)
        else:
            self.advantages = self._compute_standard_advantages()

    def end_episode(self,
                    ending_ts,
                    terminal,
                    ending_value,
                    ending_reward):
        """
        End the episode.

        Parameters:
        -----------
        ending_ts: int
            The time step we ended on.
        terminal: bool
            Did we end in a terminal state?
        ending_value: float
            The ending value of the episode.
        ending_reward: float
            The ending reward of the episode.
        """
        self.ending_ts    = ending_ts
        self.terminal     = terminal
        self.length       = self.ending_ts - self.starting_ts
        self.is_finished  = True
        self.ending_value = ending_value

        #
        # Clipping the ending value can have dramaticly positive
        # effects on training. MountainCarContinuous is a great
        # example of an environment that I've seen struggle quite
        # a bit without a propper bs clip.
        #
        if self.bootstrap_clip is not None:
            ending_reward = np.clip(
                ending_reward,
                self.bootstrap_clip[0],
                self.bootstrap_clip[1])

        padded_rewards = np.array(self.rewards + [ending_reward],
            dtype=np.float32)

        self.rewards_to_go = self.compute_discounted_sums(
            padded_rewards,
            self.gamma)[:-1]

        self.values = np.array(self.values).astype(np.float32)

        self.compute_advantages()

    def release_data(self):
        """
        Release data that is not updated in our training loop.
        """
        self.observations        = None
        self.next_observations   = None
        self.actions             = None
        self.raw_actions         = None
        self.log_probs           = None
        self.critic_observations = None

        if self.has_hidden_states:
            self.actor_hidden  = None
            self.critic_hidden = None
            self.actor_cell    = None
            self.critic_cell   = None


class AgentSharedEpisode(PPOEpisode):

    def __init__(self, agent_ids):
        """
        A container for agent shared episodes. This allows us to associate
        the episodes from agents on the same team. For instance, if a team
        contains 3 agents and we collect 5 episodes worth of experience, we
        have 15 total episodes, but they can be grouped into 5 shared episodes.

        Parameters:
        -----------
        agent_ids: array-like
            An array of agent ids on this team.
        """
        super().__init__()

        self.agent_ids      = agent_ids
        self.added_agents   = []
        self.agent_limit    = len(agent_ids)
        self.agent_count    = 0
        self.agent_episodes = np.array([None] * self.agent_limit, dtype=object)

    def verify_agent(self, agent_id):
        """
        Verify that an agent id belongs to this team.

        Parameters: 
        -----------
        agent_id: str
            The agent id to check.
        """
        if agent_id not in self.agent_ids:
            msg  = "ERROR: {agent_id} not in list of accepted "
            msg += "agents."
            rank_print(msg)
            comm.Abort()

        if agent_id in self.added_agents:
            msg  = f"ERROR: {agent_id} has already been added to this "
            msg += "AgentSharedEpisode!"
            rank_print(msg)
            comm.Abort()

    def verify_episode(self, episode):
        """
        Verify that an episode is in an appropriate state to be added to
        this container.

        Parameters:
        -----------
        episode: EpisodeInfo object.
            An instance of EpisodeInfo.
        """
        if not episode.is_finished:
            msg  = "ERROR episode must be finished before being added to and "
            msg += "AgentSharedEpisode object!"
            rank_print(msg)
            comm.Abort()

    def add_episode(self, agent_id, episode):
        """
        Add an agent's episode.

        Parameters:
        -----------
        agent_id: str
            The agent's id.
        episode: EpisodeInfo object
            The episode to add.
        """
        if self.is_finished:
            msg  = "ERROR: AgentSharedEpisode is finished! Cannot add "
            msg += "more episodes."
            rank_print(msg)
            comm.Abort()

        self.verify_agent(agent_id)
        self.verify_episode(episode)

        agent_idx = np.where(self.agent_ids == agent_id)

        self.agent_episodes[agent_idx] = episode
        self.added_agents.append(agent_id)
        self.agent_count += 1

        if self.agent_count == self.agent_limit:
            self.is_finished = True
            self._merge_episodes()

    def compute_advantages(self):
        """
        Compute the advantages of our agents.
        """
        for i in range(self.agent_limit):
            self.agent_episodes[i].values = self.values[:, i]
            self.agent_episodes[i].compute_advantages()

            self.advantages[:, i] = self.agent_episodes[i].advantages

    def _merge_episodes(self):
        """
        Merge all of our episodes so that our datasets have shapes
        of (batch_size, num_agents, data_size).
        """
        self.actions, \
        self.raw_actions, \
        self.critic_observations, \
        self.observations, \
        self.next_observations, \
        self.rewards_to_go, \
        self.log_probs, \
        ep_lens, \
        self.advantages, \
        self.values, \
        _, \
        _, \
        _, \
        _, \
        self.total_timestates = \
            combine_episodes(self.agent_episodes, False, list.append)

        #
        # Note that log_probs is a list of tensors, so we'll skip converting
        # it to a numpy array.
        #
        self.actions                  = np.stack(self.actions, axis=1)
        self.raw_actions              = np.stack(self.raw_actions, axis=1)
        self.critic_observations      = np.stack(self.critic_observations, axis=1)
        self.observations             = np.stack(self.observations, axis=1)
        self.next_observations        = np.stack(self.next_observations, axis=1)
        self.rewards_to_go            = np.stack(self.rewards_to_go, axis=1)
        self.advantages               = np.stack(self.advantages, axis=1)
        self.values                   = np.stack(self.values, axis=1)

        #
        # log_probs is a special case that needs to remain in tensor form.
        #
        self.log_probs                = [torch.cat(lp) for lp in self.log_probs]
        self.log_probs                = torch.stack(self.log_probs, axis=1).to(torch.float32)

        if len(ep_lens) == 0:
            msg  = "ERROR: attempting to merge AgentSharedEpisode with "
            msg += "empty episodes!"
            rank_print(msg)
            comm.Abort()

        self.length = ep_lens[0]

        if (ep_lens != self.length).any():
            msg  = "ERROR: mismatched episodes found in AgentSharedEpisode! "
            msg += f"Episode lengths: {ep_lens}"
            rank_print(msg)
            comm.Abort()

    def release_data(self):
        """
        Release data that is not updated in our training loop.
        """
        for ep in self.agent_episodes:
            ep.release_data()


class PPODataset(Dataset):

    def __init__(self,
                 device,
                 action_dtype,
                 sequence_length = 1):
        """
        A PyTorch Dataset representing our rollout data.

        Parameters:
        -----------
        device: torch.device
            The device we're training on.
        action_dtype: str
            The action data dtype (discrete/continuous).
        sequence_length: int
            If set to > 1, our dataset will return
            obervations as sequences of this length.
        """

        self.action_dtype         = action_dtype
        self.device               = device
        self.episodes             = []
        self.is_built             = False
        self.build_hidden_states  = False
        self.sequence_length      = sequence_length
        self.build_terminal_mask  = False
        self.shared               = False

        if self.sequence_length <= 0:
            msg  = "ERROR: PPODataset must have a sequence length >= 1 "
            msg += "but received {}".format(self.sequence_length)
            rank_print(msg)
            comm.Abort()

        elif self.sequence_length > 1:
            self.build_terminal_mask = True

        self.actions                   = None
        self.raw_actions               = None
        self.critic_observations       = None
        self.observations              = None
        self.next_observations         = None
        self.rewards_to_go             = None
        self.log_probs                 = None
        self.ep_lens                   = None
        self.advantages                = None
        self.actor_hidden              = None
        self.critic_hidden             = None
        self.actor_cell                = None
        self.critic_cell               = None
        self.terminal_mask             = None
        self.values                    = None

    def add_episode(self, episode):
        """
        Add an episode to our dataset.

        Parameters:
        -----------
        episode: PPOEpisode
            The episode to add.
        """
        if episode.has_hidden_states:
            self.build_hidden_states = True

        elif self.build_hidden_states  and not episode.has_hidden_states:
            msg  = "ERROR: some episodes have hidden states while others "
            msg += "do not. Bailing..."
            rank_print(msg)
            comm.Abort()

        self.episodes.append(episode)

    def recalculate_advantages(self):
        """
        Recalculate our advantages. This can be used to mitigate using
        stale advantages when training over > 1 epochs.
        """
        if not self.is_built:
            msg  = "WARNING: recalculate_advantages was called before "
            msg += "the dataset has been built. Ignoring call."
            rank_print(msg)
            return

        val_idx = 0
        for ep in self.episodes:
            for ep_ts in range(ep.length):
                ep.values[ep_ts] = self.values[val_idx]
                val_idx += 1

            ep.compute_advantages()

        self.advantages = combine_episode_advantages(self.episodes)
        self.advantages = np.array(self.advantages)
        self.advantages = torch.tensor(self.advantages,
            dtype=torch.float32).to(self.device)

    def build(self):
        """
        Build our dataset from the existing episodes.
        """

        if self.is_built:
            msg  = "ERROR: attempting to build a batch, but it's "
            msg += "already been built! Bailing..."
            rank_print(msg)
            comm.Abort()

        self.num_episodes = len(self.episodes)

        self.actions, \
        self.raw_actions, \
        self.critic_observations, \
        self.observations, \
        self.next_observations, \
        self.rewards_to_go, \
        self.log_probs, \
        self.ep_lens, \
        self.advantages, \
        self.values, \
        self.actor_hidden, \
        self.critic_hidden, \
        self.actor_cell, \
        self.critic_cell, \
        self.total_timestates = \
            combine_episodes(self.episodes, self.build_hidden_states)

        if self.build_terminal_mask:
            terminal_mask = np.zeros(self.total_timestates).astype(np.bool_)
            cur_ts = 0

        if self.build_terminal_mask:
            for ep in self.episodes:

                if not ep.is_finished:
                    msg  = "ERROR: attempting to build a batch using "
                    msg += "an incomplete episode! Bailing..."
                    rank_print(msg)
                    comm.Abort()

                cur_ts += ep.length

                if ep.terminal:
                    terminal_mask[cur_ts - 1] = True

            max_ts = self.total_timestates - (self.sequence_length - 1)
            self.terminal_sequence_masks = np.array(
                [None] * max_ts, dtype=object)

            for ts in range(max_ts):
                mask     = np.zeros(self.sequence_length).astype(np.bool_)
                mask_idx = 0
                stop_ts  = ts + self.sequence_length

                for mask_ts in range(ts, stop_ts):
                    if terminal_mask[mask_ts]: 
                        mask[mask_idx + 1:] = True
                        break

                    mask_idx += 1

                self.terminal_sequence_masks[ts] = mask

        #
        # Note that log_probs is a list of tensors, so we'll skip converting
        # it to a numpy array.
        #
        self.actions              = np.array(self.actions)
        self.raw_actions          = np.array(self.raw_actions)
        self.critic_observations  = np.array(self.critic_observations)
        self.observations         = np.array(self.observations)
        self.next_observations    = np.array(self.next_observations)
        self.rewards_to_go        = np.array(self.rewards_to_go)
        self.ep_lens              = np.array(self.ep_lens)
        self.advantages           = np.array(self.advantages)
        self.values               = torch.tensor(self.values, dtype=torch.float32)

        self.values               = self.values.to(self.device)

        if self.total_timestates != len(self.observations):
            error_msg  = "ERROR: expected the total timestates to match "
            error_msg += "the total number of observations, but got "
            error_msg += "{} vs {}".format(self.total_timestates,
                self.observations.shape)
            rank_print(error_msg)
            comm.Abort()

        if self.build_hidden_states:
            #
            # Torch expects the shape to be
            # (num_lstm_layers, batch_size, hidden_size), but our
            # data loader will concat on the first dimension. So, we
            # need to transpose so that the batch size comes first.
            #
            self.actor_hidden = torch.tensor(
                np.concatenate(self.actor_hidden, axis=1), dtype=torch.float32).to(self.device)

            self.critic_hidden = torch.tensor(
                np.concatenate(self.critic_hidden, axis=1), dtype=torch.float32).to(self.device)

            self.actor_cell = torch.tensor(
                np.concatenate(self.actor_cell, axis=1), dtype=torch.float32).to(self.device)

            self.critic_cell = torch.tensor(
                np.concatenate(self.critic_cell, axis=1), dtype=torch.float32).to(self.device)

            self.actor_hidden  = torch.transpose(self.actor_hidden, 0, 1)
            self.actor_cell    = torch.transpose(self.actor_cell, 0, 1)
            self.critic_hidden = torch.transpose(self.critic_hidden, 0, 1)
            self.critic_cell   = torch.transpose(self.critic_cell, 0, 1)

        else:
            empty_state = np.zeros(self.total_timestates).astype(np.uint8)

            self.actor_hidden  = empty_state
            self.critic_hidden = empty_state

            self.actor_cell    = empty_state
            self.critic_cell   = empty_state

        self.advantages = torch.tensor(self.advantages,
            dtype=torch.float32).to(self.device)

        self.observations = torch.tensor(self.observations,
            dtype=torch.float32).to(self.device)

        self.next_observations = torch.tensor(self.next_observations,
            dtype=torch.float32).to(self.device)

        self.critic_observations = torch.tensor(self.critic_observations,
            dtype=torch.float32).to(self.device)

        if self.shared:
            self.log_probs = torch.stack(self.log_probs).to(torch.float32).to(self.device)
        else:
            self.log_probs = torch.tensor(self.log_probs,
                dtype=torch.float32).to(self.device)

        self.rewards_to_go = torch.tensor(self.rewards_to_go,
            dtype=torch.float32).to(self.device)

        if self.action_dtype in ["continuous", "multi-binary", "mixed"]:
            self.actions = torch.tensor(self.actions,
                dtype=torch.float32).to(self.device)

            self.raw_actions = torch.tensor(self.raw_actions,
                dtype=torch.float32).to(self.device)

        elif self.action_dtype in ["discrete", "multi-discrete"]:
            self.actions = torch.tensor(self.actions,
                dtype=torch.long).to(self.device)
            self.raw_actions = torch.tensor(self.raw_actions,
                dtype=torch.long).to(self.device)

        else:
            msg  = f"ERROR: unknown action_dtype  of {self.action_dtype} encountered "
            msg += "in PPODataset."
            rank_print(msg)
            comm.Abort()

        if len(self.actions.shape) <= 1:
            self.actions = torch.unsqueeze(self.actions, dim=-1)

        if len(self.raw_actions.shape) <= 1:
            self.raw_actions = torch.unsqueeze(self.raw_actions, dim=-1)

        self.is_built = True

    def __len__(self):
        """
        Get the length of our dataset.
        """
        return self.total_timestates - (self.sequence_length - 1)

    def __getitem__(self, idx):
        """
        Get data from the requested index.

        Parameters:
        -----------
        idx: int
            The index to retrieve data from.

        Returns:
        --------
        tuple
            All data associated with the given index in our dataset.
        """
        #
        # First, handle the easy case of using single time states.
        #
        if self.sequence_length == 1:
            return (self.critic_observations[idx],
                    self.observations[idx],
                    self.next_observations[idx],
                    self.raw_actions[idx],
                    self.actions[idx],
                    self.advantages[idx],
                    self.log_probs[idx],
                    self.rewards_to_go[idx],
                    self.actor_hidden[idx],
                    self.critic_hidden[idx],
                    self.actor_cell[idx],
                    self.critic_cell[idx],
                    idx)

        #
        # We want our time sequence to start at the most recent state,
        # which means we'll be building a "tail" of history samples.
        #
        idx  += (self.sequence_length - 1)
        start = idx - (self.sequence_length - 1)
        stop  = idx + 1

        glob_obs_seq = self.critic_observations[start : stop].clone()
        obs_seq      = self.observations[start : stop].clone()
        nxt_obs_seq  = self.next_observations[start : stop].clone()

        #
        # Once we hit a terminal state, all subsequent observations
        # need to be zero'd out so that the model can learn that future
        # observations don't exist in this trajectory.
        #
        term_mask              = self.terminal_sequence_masks[start]
        obs_seq[term_mask]     = 0.0
        nxt_obs_seq[term_mask] = 0.0

        return (glob_obs_seq,
                obs_seq,
                nxt_obs_seq,
                self.raw_actions[idx],
                self.actions[idx],
                self.advantages[idx],
                self.log_probs[idx],
                self.rewards_to_go[idx],
                self.actor_hidden[idx],
                self.critic_hidden[idx],
                self.actor_cell[idx],
                self.critic_cell[idx],
                idx)


class PPOSharedEpisodeDataset(PPODataset):

    def __init__(
        self,
        num_envs,
        agent_ids,
        *args,
        **kw_args):
        """
        A version of our PPODataset that uses AgentSharedEpisodes.

        Parameters:
        -----------
        num_envs: int
            How many environments are we tracking on this rank?
        agent_ids: array-like
            An array of our agent ids.
        """

        super(PPOSharedEpisodeDataset, self).__init__(*args, **kw_args)

        self.num_envs      = num_envs
        self.agent_ids     = agent_ids
        self.shared        = True

        #
        # Construct a "queue" of agent shared episodes.
        #
        shared_episodes = []
        for _ in range(num_envs):
            shared_episodes.append(AgentSharedEpisode(agent_ids))

        self.episode_queue = np.array(shared_episodes)

    def add_shared_episode(self, episode, agent_id, env_idx):
        """
        Add a shared episode from a single agent to our episode queue.
        Agents that share episodes will put their episodes all into the
        same AgentSharedEpisode object.

        Parameters:
        -----------
        episode: EpisodeInfo object
            An completed EpisodeInfo object.
        agent_id: str
            The associated agent's id.
        env_idx: int
            Which environment did this come from?
        """
        self.episode_queue[env_idx].add_episode(agent_id, episode)

        #
        # If we've added all agent episodes, the shared episode will be
        # flagged as finished. We then add the shared episode to our
        # list of general episodes and replace its spot in the queue with
        # a new shared episode.
        #
        if self.episode_queue[env_idx].is_finished:

            self.episodes.append(self.episode_queue[env_idx])
            self.episode_queue[env_idx] = AgentSharedEpisode(self.agent_ids)

    def __len__(self):
        """
        Get the length of our dataset.
        """
        return self.total_timestates

    def __getitem__(self, idx):
        """
        Shuffle our agents and return data.

        Parameters:
        -----------
        idx: int
            The index to retrieve data from.

        Returns:
        --------
        tuple
            All data associated with the given index in our dataset.
        """
        return (self.critic_observations[idx],
                self.observations[idx],
                self.next_observations[idx],
                self.raw_actions[idx],
                self.actions[idx],
                self.advantages[idx],
                self.log_probs[idx],
                self.rewards_to_go[idx],
                self.actor_hidden[idx],
                self.critic_hidden[idx],
                self.actor_cell[idx],
                self.critic_cell[idx],
                idx)
