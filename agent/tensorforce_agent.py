# tensorforce_agent.py
from tensorforce import Agent
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

def get_dueling_dqn_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    config = {
        "agent": "dqn",
        "network": model,  # or 'auto'
        "memory": 10000,
        "update_frequency": 1,
        "learning_rate": learning_rate,
        "exploration": 0.1,
        "batch_size": 32,
        "target_update_frequency": 100,
        "max_episode_timesteps": max_episode_timesteps,
        "device": device
    }

    return Agent.create(
        config=config,
        environment=environment
    )
