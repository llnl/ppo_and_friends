import gymnasium as gym
from ppo_and_friends.environments.gym.wrappers import SingleAgentGymWrapper
from ppo_and_friends.policies.utils import get_single_policy_defaults
from ppo_and_friends.runners.env_runner import GymRunner
from ppo_and_friends.networks.ppo_networks.feed_forward import FeedForwardNetwork
from ppo_and_friends.utils.schedulers import *
import torch.nn as nn
from ppo_and_friends.environments.action_wrappers import AlternateActionEnvWrapper
from gymnasium.spaces import MultiBinary
from ppo_and_friends.runners.runner_tags import ppoaf_runner

class MultiBinaryLunarLanderWrapper(AlternateActionEnvWrapper):
    """
        A simple multi-binary action version of LunarLander. This is for
        testing purposes only.
    """

    def _get_action_space(self):
        return MultiBinary(2)

    def step(self, action):

        step_action = None

        action = action.flatten()

        if action.sum() == 0:
           step_action = 0
        elif action.sum() == 2:
            step_action = 1
        elif action[0] == 1 and action[1] == 0:
            step_action = 2
        elif action[0] == 0 and action[1] == 1:
            step_action = 3

        return self.env.step(step_action)

@ppoaf_runner
class BinaryLunarLanderRunner(GymRunner):
    """
        This is merely the original LunarLander environment wrapped in a
        MultiBinary action space. It's used to test our multi-binary
        action space support.
    """

    def run(self):
        env_generator = lambda : \
            SingleAgentGymWrapper(
                MultiBinaryLunarLanderWrapper(gym.make('LunarLander-v3',
                render_mode = self.get_gym_render_mode())))

        #
        # Extra args for the actor critic models.
        # I find that leaky relu does much better with the lunar
        # lander env.
        #
        actor_kw_args = {}

        actor_kw_args["activation"]  = nn.LeakyReLU()
        actor_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()
        critic_kw_args["hidden_size"] = 256

        critic_kw_args = actor_kw_args.copy()

        lr = LinearScheduler(
            status_key    = "iteration",
            status_max    = 200,
            max_value     = 0.0003,
            min_value     = 0.0001)

        policy_args = {\
            "ac_network"       : FeedForwardNetwork,
            "actor_kw_args"    : actor_kw_args,
            "critic_kw_args"   : critic_kw_args,
            "lr"               : lr,
        }

        policy_settings, policy_mapping_fn = get_single_policy_defaults(
            env_generator = env_generator,
            policy_args   = policy_args)

        self.run_ppo(env_generator       = env_generator,
                     policy_settings     = policy_settings,
                     policy_mapping_fn   = policy_mapping_fn,
                     max_ts_per_ep       = 512,
                     ts_per_rollout      = 1024,
                     batch_size          = 512,
                     normalize_obs       = True,
                     normalize_rewards   = True,
                     obs_clip            = (-10., 10.),
                     reward_clip         = (-10., 10.),
                     **self.kw_run_args)
