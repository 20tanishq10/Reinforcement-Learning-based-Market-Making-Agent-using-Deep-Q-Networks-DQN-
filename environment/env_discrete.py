import numpy as np
import os
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box


class EnvDiscrete(Env):
    def __init__(self, code, day, latency, T=50, wo_lob_state=False, wo_market_state=False,
                 wo_dampened_pnl=False, wo_matched_pnl=False, wo_inv_punish=False,
                 experiment_name='', log=False):

        self.code = code
        self.day = day
        self.T = T
        self.latency = latency
        self.log = log

        self.episode_idx = 0
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_dampened_pnl = wo_dampened_pnl
        self.wo_matched_pnl = wo_matched_pnl
        self.wo_inv_punish = wo_inv_punish

        # Load processed data
        npz = np.load(f'./processed/{day}.npz')
        self.lob = npz['lob']
        self.market = npz['market']

        # Define discrete action space: [hold, buy, sell]
        self.action_space = Discrete(3)

        # State space is flattened LOB + market info
        lob_shape = self.lob.shape[1:] if not self.wo_lob_state else (0, 15 * 4)
        market_shape = self.market.shape[1:] if not self.wo_market_state else (0,)
        obs_dim = np.prod(lob_shape) + np.prod(market_shape)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset_seq(self, timesteps_per_episode, episode_idx):
        self.timestep = 0
        self.episode_idx = episode_idx
        self.timesteps_per_episode = timesteps_per_episode
        self.start = episode_idx * timesteps_per_episode
        return self._get_obs()

    def reset(self):
        self.timestep = 0
        self.start = 0
        return self._get_obs()

    def _get_obs(self):
        idx = self.start + self.timestep
        lob_state = self.lob[idx].flatten() if not self.wo_lob_state else np.array([])
        market_state = self.market[idx] if not self.wo_market_state else np.array([])
        obs = np.concatenate([lob_state, market_state])
        return obs.astype(np.float32)

    def execute(self, actions):
        # Action: 0 = hold, 1 = buy, 2 = sell
        reward = 0
        done = False
        self.timestep += 1
        next_obs = self._get_obs()

        if self.timestep >= self.timesteps_per_episode:
            done = True

        # Dummy reward logic (to be replaced with your logic)
        reward = np.random.randn()

        return next_obs, done, reward

    def get_final_result(self):
        # Dummy metrics, to be replaced with real evaluation logic
        return {
            'pnl': np.random.uniform(0, 1),
            'nd_pnl': np.random.uniform(0, 1),
            'avg_abs_position': np.random.uniform(0, 1),
            'profit_ratio': np.random.uniform(0, 1),
            'volume': np.random.uniform(0, 1)
        }
