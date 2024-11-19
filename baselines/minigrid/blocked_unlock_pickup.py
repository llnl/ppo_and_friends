import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
import minigrid

@ppoaf_runner
class BlockedUnlockPickupRunner(GymRunner):

    def add_cli_args(self, parser):
        """
        Define extra args that will be added to the ppoaf command.

        Parameters:
        -----------
        parser: argparse.ArgumentParser
            The parser from ppoaf.

        Returns:
        --------
        argparse.ArgumentParser:
            The same parser as the input with potentially new arguments added.
        """
        parser.add_argument("--bs_clip_min", default=-np.inf, type=float)
        parser.add_argument("--bs_clip_max", default=np.inf, type=float)

        parser.add_argument("--max_env_steps", default=512, type=int)

        parser.add_argument("--ts_per_rollout", default=1024, type=int)
        parser.add_argument("--max_ts_per_ep", default=32, type=int)
        parser.add_argument("--epochs_per_iter", default=32, type=int)

        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--actor_hidden", type=int, default=32)
        parser.add_argument("--critic_hidden_mult", type=int, default=2)

        #
        # ICM args.
        #
        parser.add_argument("--enable_icm", type=int, default=0, choices=[0,1])
        parser.add_argument("--icm_inverse_size", type=int, default=32)
        parser.add_argument("--icm_inverse_depth", type=int, default=2)
        parser.add_argument("--icm_forward_size", type=int, default=32)
        parser.add_argument("--icm_forward_depth", type=int, default=2)
        parser.add_argument("--icm_encoded_obs_dim", type=int, default=2)
        parser.add_argument("--icm_learning_rate", type=float, default=0.001)
        parser.add_argument("--intr_reward_weight", type=float, default=100.0)
        return parser

    def run(self):

        env_generator = lambda : \
            SingleAgentGymWrapper(
                gym.make(
                    'MiniGrid-BlockedUnlockPickup-v0',
                    render_mode = self.get_gym_render_mode(),
                    max_steps   = self.cli_args.max_env_steps))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = self.cli_args.actor_hidden

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = self.cli_args.actor_hidden * self.cli_args.critic_hidden_mult

        icm_kw_args = {}
        icm_kw_args["encoded_obs_dim"]      = self.cli_args.icm_encoded_obs_dim
        icm_kw_args["inverse_hidden_depth"] = self.cli_args.icm_inverse_depth
        icm_kw_args["inverse_hidden_size"]  = self.cli_args.icm_inverse_size
        icm_kw_args["forward_hidden_depth"] = self.cli_args.icm_forward_depth
        icm_kw_args["forward_hidden_size"]  = self.cli_args.icm_forward_size

        bs_clip_min = self.cli_args.bs_clip_min
        bs_clip_max = self.cli_args.bs_clip_max

        policy_args = {\
            "ac_network"         : FeedForwardNetwork,
            "actor_kw_args"      : actor_kw_args,
            "critic_kw_args"     : critic_kw_args,
            "lr"                 : self.cli_args.learning_rate,
            "enable_icm"         : self.cli_args.enable_icm,
            "icm_kw_args"        : icm_kw_args,
            "icm_lr"             : self.cli_args.icm_learning_rate,
            "bootstrap_clip"     : (bs_clip_min, bs_clip_max),
            "intr_reward_weight" : self.cli_args.intr_reward_weight,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     ts_per_rollout     = self.cli_args.ts_per_rollout,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = self.cli_args.max_ts_per_ep,
                     epochs_per_iter    = self.cli_args.epochs_per_iter,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     normalize_values   = False,
                     obs_clip           = None,
                     reward_clip        = None,
                     **self.kw_run_args)
