import gymnasium as gym
from ppo_and_friends.environments.minigrid.wrappers import SingleAgentMiniGridWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.runners.runner_tags import ppoaf_runner
import minigrid

@ppoaf_runner
class BlockedUnlockPickupRunner(GymRunner):

    def run(self):

        env_generator = lambda : \
            SingleAgentMiniGridWrapper(
                gym.make(
                    'MiniGrid-BlockedUnlockPickup-v0',
                    render_mode = self.get_gym_render_mode(),
                    max_steps   = 512))

        actor_kw_args = {}
        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 128

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        lr = 0.0003

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
            #"bootstrap_clip"   : (20.0, 100),
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator      = env_generator,
                     ts_per_rollout     = 1024,
                     policy_settings    = policy_settings,
                     policy_mapping_fn  = policy_mapping_fn,
                     max_ts_per_ep      = 128,
                     epochs_per_iter    = 16,
                     obs_clip           = None,
                     reward_clip        = None,
                     normalize_obs      = False,
                     normalize_rewards  = False,
                     **self.kw_run_args)
