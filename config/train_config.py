# config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    code: str = 'demo'
    device: str = 'cpu'
    latency: int = 1
    time_window: int = 50
    log: bool = False
    exp_name: str = ''
    agent_type: str = 'ppo'  # or 'dueling_dqn'
    learning_rate: float = 1e-4
    horizon: int = 1
    env_type: str = 'discrete'  # or 'continuous'
    load: bool = False
    agent_load_dir: str = ''
    save: bool = False
    agent_save_dir: str = ''

    # Ablation flags
    wo_pretrain: bool = False
    wo_attnlob: bool = False
    wo_lob_state: bool = False
    wo_market_state: bool = False
    wo_dampened_pnl: bool = False
    wo_matched_pnl: bool = False
    wo_inv_punish: bool = False
