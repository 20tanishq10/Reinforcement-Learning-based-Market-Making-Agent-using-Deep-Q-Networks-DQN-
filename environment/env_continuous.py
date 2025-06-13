import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from utils.data_processing import load_orderbook_data, load_trade_data


class EnvContinuous(Env):
    def __init__(self, code, day, latency=1, T=50,
                 wo_lob_state=False, wo_market_state=False,
                 wo_dampened_pnl=False, wo_matched_pnl=False, wo_inv_punish=False,
                 experiment_name='', log=False):

        super(EnvContinuous, self).__init__()

        self.code = code
        self.day = day
        self.latency = latency
        self.T = T
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_dampened_pnl = wo_dampened_pnl
        self.wo_matched_pnl = wo_matched_pnl
        self.wo_inv_punish = wo_inv_punish
        self.experiment_name = experiment_name
        self.log = log

        #self.orderbook = load_orderbook_data(f"data/orderbook/{code}_{day}.csv")
        #self.trade = load_trade_data(f"data/trades/{code}_{day}.csv")
        data = np.load(f"./processed/{day}.npz")
        self.orderbook = data['lob']         # shape (N, 40)
        self.trade = data['market']          # shape (N, 5)


        self.max_position = 100
        self.inventory = 0
        self.pnl = 0
        self.current_step = 0

        # Action space: continuous [-1, 1] range (for order placement decisions)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # State space: dynamic shape based on config
        self.state_dim = self._get_state().shape[0]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def _get_state(self):
        # Slice LOB state (bids/asks with depth=10) and normalize
        #lob_features = self.orderbook.iloc[self.current_step]
        #lob_values = lob_features.filter(regex='bids_|asks_').values[:20]  # 10 each

        #market_features = self.trade.iloc[self.current_step][['Volume', 'Open', 'High', 'Low', 'Close']].values

        lob_values = self.orderbook[self.current_step][:20]  # First 20: 10 bids + 10 asks
        market_features = self.trade[self.current_step]      # Already a 1D array with 5 elements


        state = []
        if not self.wo_lob_state:
            state.extend(lob_values)
        if not self.wo_market_state:
            state.extend(market_features)

        return np.array(state, dtype=np.float32)

    def reset(self):
        self.inventory = 0
        self.pnl = 0
        self.current_step = self.T
        return self._get_state()

    def reset_seq(self, timesteps_per_episode=2000, episode_idx=0):
        self.episode_idx = episode_idx
        self.current_step = episode_idx * timesteps_per_episode + self.T
        return self._get_state()

    def execute(self, actions):
        action = actions[0]  # Single continuous action [-1, 1]

        midpoint = self.orderbook.iloc[self.current_step]['midpoint']
        spread = self.orderbook.iloc[self.current_step]['spread']

        price = midpoint + action * spread / 2
        executed = np.random.choice([1, -1])  # Mock: 1 buy, -1 sell

        volume = self.trade.iloc[self.current_step]['Volume']
        position_change = executed
        reward = -abs(price - midpoint) * volume  # mock reward as spread capture

        if not self.wo_dampened_pnl:
            reward *= 0.95

        self.inventory += position_change
        self.pnl += reward

        self.current_step += 1
        terminal = self.current_step >= len(self.orderbook) - 1

        return self._get_state(), terminal, reward

    @property
    def states(self):
        return dict(type='float', shape=self.observation_space.shape)

    @property
    def actions(self):
        return dict(type='float', shape=self.action_space.shape)

    def get_final_result(self):
        return {
            'pnl': self.pnl,
            'nd_pnl': self.pnl / (abs(self.inventory) + 1e-5),
            'avg_abs_position': abs(self.inventory),
            'profit_ratio': self.pnl / (abs(self.pnl) + abs(self.inventory) + 1e-5),
            'volume': self.trade.iloc[:self.current_step]['Volume'].sum()
        }
