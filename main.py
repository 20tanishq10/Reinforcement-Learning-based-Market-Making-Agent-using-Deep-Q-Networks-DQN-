# main.py
import os
import pandas as pd
from tqdm import tqdm
from dataclasses import asdict
from tensorflow import keras
import pyrallis
from config import TrainConfig
#from environment.env_discrete import EnvDiscrete
from environment.env_continuous import EnvContinuous
from agent.tensorforce_agent import get_dueling_dqn_agent, get_ppo_agent
from network.network import get_model, get_lob_model, get_fclob_model, get_pretrain_model, compute_output_shape

from tensorforce.environments import Environment
from environment.env_continuous import EnvContinuous

train_days = ['20210407', '20210408', '20210409', '20210410' ,'20210411', '20210412'
 '20210413' ,'20210414', '20210415', '20210416']
test_days = ['20210417', '20210418', '20210419']
num_step_per_episode = 200
n_train_loop = 5
keras_model_dir = './keras_model'

def init_env(day, config):
    env_cls = EnvContinuous #if config['env_type'] == 'continuous' else EnvDiscrete
    gym_env = env_cls(
        code=config['code'],
        day=day,
        latency=config['latency'],
        T=config['time_window'],
        wo_lob_state=config['wo_lob_state'],
        wo_market_state=config['wo_market_state'],
        wo_dampened_pnl=config['wo_dampened_pnl'],
        wo_matched_pnl=config['wo_matched_pnl'],
        wo_inv_punish=config['wo_inv_punish'],
        experiment_name=config['exp_name'],
        log=config['log']
    )
    # ✅ Wrap it here
    return Environment.create(environment=gym_env)

def init_agent(env, config):
    kwargs = {
        'learning_rate': config['learning_rate'],
        'horizon': config['horizon']
    }
    get_agent = get_ppo_agent if config['agent_type'] == 'ppo' else get_dueling_dqn_agent

    if config['wo_pretrain']:
        lob_model = get_lob_model(64, config['time_window'])
        lob_model.compute_output_shape = compute_output_shape
    else:
        base_model = get_lob_model(64, config['time_window'])
        model_pretrain = get_pretrain_model(base_model, config['time_window'])
        model_pretrain.load_weights(f'./ckpt/pretrain_model_{config["code"]}/weights')
        lob_model = model_pretrain.layers[0]

    if config['wo_attnlob']:
        lob_model = get_fclob_model(64, config['time_window'])

    model = get_model(
        lob_model,
        config['time_window'],
        with_lob_state=not config['wo_lob_state'],
        with_market_state=not config['wo_market_state']
    )
    agent = get_agent(environment=env, max_episode_timesteps=1000, device=config['device'], **kwargs)

    if config['load']:
        model = keras.models.load_model(keras_model_dir)
        model.layers[0].compute_output_shape = compute_output_shape
        agent = get_agent(environment=env, max_episode_timesteps=1000, device=config['device'], **kwargs)
        agent.restore(config['agent_load_dir'], filename='cppo', format='numpy')

    return agent
    print(agent.states)
    print(agent.actions)
    print(agent)


def train_a_day(env, agent, train_result):
    num_episodes = len(env.orderbook) // num_step_per_episode
    for idx in tqdm(range(num_episodes)):
        states = env.reset_seq(num_step_per_episode, episode_idx=idx)
        terminal = False
        ep_states, ep_actions, ep_terminal, ep_rewards = [], [], [], []
        while not terminal:
            ep_states.append(states)
            action = agent.act(states=states, independent=True)
            ep_actions.append(action)
            states, terminal, reward = env.execute(actions=action)
            ep_terminal.append(terminal)
            ep_rewards.append(reward)

        agent.experience(states=ep_states, actions=ep_actions, terminal=ep_terminal, reward=ep_rewards)
        agent.update()
        save_episode_result(env, train_result)

def test_a_day(env, agent, test_result):
    num_episodes = len(env.orderbook) // num_step_per_episode
    for idx in tqdm(range(num_episodes)):
        states = env.reset_seq(num_step_per_episode, episode_idx=idx)
        terminal = False
        while not terminal:
            action = agent.act(states=states, independent=True)
            states, terminal, _ = env.execute(actions=action)
        save_episode_result(env, test_result)

def train(agent, train_result, config):
    for day in train_days:
        env = init_env(day, config)
        train_a_day(env, agent, train_result)

def test(agent, test_result, config):
    for day in test_days:
        env = init_env(day, config)
        test_a_day(env, agent, test_result)

def save_episode_result(env, result_df):
    res = env.get_final_result()
    key = f"{env.day}_{env.episode_idx}"
    result_df.loc[key] = [res['pnl'], res['nd_pnl'], res['avg_abs_position'], res['profit_ratio'], res['volume']]

def gather_test_results(result_df):
    grouped = result_df.groupby(result_df.index.str[:10])
    summary = grouped.agg({
        'PnL': 'sum',
        'ND-PnL': 'sum',
        'average_position': 'mean',
        'profit_ratio': 'mean',
        'volume': 'sum'
    })
    return summary.sort_index()

def save_agent(agent, config):
    agent.model.policy.network.keras_model.save(keras_model_dir)
    agent.save(config['agent_save_dir'], filename='cppo', format='numpy')

@pyrallis.wrap()
def main(config: TrainConfig):
    config = asdict(config)
    env = init_env(train_days[0], config)
    agent = init_agent(env, config)

    train_result = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    for _ in range(n_train_loop):
        train(agent, train_result, config)
        if config['save']:
            save_agent(agent, config)

    test_result = pd.DataFrame(columns=['PnL', 'ND-PnL', 'average_position', 'profit_ratio', 'volume'])
    test(agent, test_result, config)
    daily_results = gather_test_results(test_result)
    print(daily_results)

if __name__ == '__main__':
    main()
