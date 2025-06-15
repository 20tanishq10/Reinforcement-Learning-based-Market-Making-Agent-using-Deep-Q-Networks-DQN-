# tensorforce_agent.py
from tensorforce.agents import PPOAgent, DQNAgent

def get_ppo_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    return PPOAgent(
        states=environment.states,
        actions=environment.actions,
        network='auto',
        batch_size=10,
        learning_rate=1e-3,
        update_frequency=10,
        max_episode_timesteps=2000
        )

from tensorforce.agents import DQNAgent

def get_dueling_dqn_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    return DQNAgent(
        states=environment.states,
        actions=environment.actions,
        network=model,
        max_episode_timesteps=max_episode_timesteps,
        update=dict(unit='timesteps', batch_size=32),
        memory=10000,
        optimizer=dict(type='adam', learning_rate=learning_rate),
        exploration=dict(
            type='linear', unit='timesteps', num_steps=10000,
            initial_value=1.0, final_value=0.1
        ),
        device=device
    )


