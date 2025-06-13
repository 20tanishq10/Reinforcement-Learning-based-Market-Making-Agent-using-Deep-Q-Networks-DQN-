# env_discrete.py
import numpy as np
import os
import gym
from gym import spaces

class EnvDiscrete(gym.Env):
    def __init__(self, code, day, latency=1, T=50,
                 wo_lob_state=False, wo_market_state=False,
                 wo_dampened_pnl=False, wo_matched_pnl=False,
                 wo_inv_punish=False,
                 experiment_name='', log=False):

        super(EnvDiscrete, self).__init__()

        self.code = code
        self.day = day
        self.latency = latency
        self.T = T
        self.log = log
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_dampened_pnl = wo_dampened_pnl
        self.wo_matched_pnl = wo_matched_pnl
        self.wo_inv_punish = wo_inv_punish
        self.experiment_name = experiment_name

        self.episode_idx = 0
        self.timesteps_per_episode = 2000

        # Load data from preprocessed .npz file
        self.data_path = f'./processed/{day}.npz'
        data = np.load(self.data_path)
        self.lob_data = data['lob']
        self.market_data = data['market']

        self.max_episode = len(self.lob_data) // self.timesteps_per_episode

        # Action space: discrete values (e.g., Buy, Sell, Hold)
        self.action_space = spaces.Discrete(3)

        # Observation space: flatten lob + market features
        lob_shape = self.lob_data[0].shape  # (T, features)
        market_shape = self.market_data[0].shape  # (features,)

        obs_dim = 0
        if not self.wo_lob_state:
            obs_dim += lob_shape[0] * lob_shape[1]
        if not self.wo_market_state:
            obs_dim += market_shape[0]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.reset_seq(self.timesteps_per_episode, self.episode_idx)

    def reset_seq(self, timesteps_per_episode, episode_idx):
        self.timesteps_per_episode = timesteps_per_episode
        self.episode_idx = episode_idx
        self.current_step = 0

        start = episode_idx * timesteps_per_episode
        end = (episode_idx + 1) * timesteps_per_episode

        self.lob_episode = self.lob_data[start:end]
        self.market_episode = self.market_data[start:end]

        return self._get_obs()

    def reset(self):
        return self.reset_seq(self.timesteps_per_episode, self.episode_idx)

    def _get_obs(self):
        obs_parts = []
        if not self.wo_lob_state:
            obs_parts.append(self.lob_episode[self.current_step].flatten())
        if not self.wo_market_state:
            obs_parts.append(self.market_episode[self.current_step])
        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)

    def execute(self, actions):
        reward = self._compute_reward(actions)

        self.current_step += 1
        terminal = self.current_step >= self.timesteps_per_episode
        next_state = self._get_obs() if not terminal else np.zeros_like(self._get_obs())

        return next_state, terminal, reward

    def _compute_reward(self, action):
        # Simplified reward: +1 for Buy, -1 for Sell, 0 for Hold
        # Replace with custom reward logic
        if action == 0:  # Buy
            return 1.0
        elif action == 1:  # Sell
            return -1.0
        else:  # Hold
            return 0.0

    @property
    def states(self):
        return dict(type='float', shape=self.observation_space.shape)

    @property
    def actions(self):
        return dict(type='int', num_actions=self.action_space.n)

    def get_final_result(self):
        # Dummy implementation — replace with actual result computation
        return {
            'pnl': 0,
            'nd_pnl': 0,
            'avg_abs_position': 0,
            'profit_ratio': 1,
            'volume': 1,
        }

