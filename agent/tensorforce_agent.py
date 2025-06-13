# tensorforce_agent.py
from tensorforce.agents import PPOAgent, DQNAgent

def get_ppo_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    return PPOAgent(
        states=environment.states,
        actions=environment.actions(),
        max_episode_timesteps=max_episode_timesteps,
        network=model,
        update_mode=dict(unit='episodes', batch_size=10),
        optimizer=dict(optimizer='adam', learning_rate=learning_rate),
        objective='policy_gradient',
        reward_estimation=dict(horizon=horizon),
        exploration=0.1,
        device=device
    )

def get_dueling_dqn_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    return DQNAgent(
        states=environment.states,
        actions=environment.actions(),
        max_episode_timesteps=max_episode_timesteps,
        network=model,
        update_mode=dict(unit='timesteps', batch_size=32),
        memory=10000,
        optimizer=dict(optimizer='adam', learning_rate=learning_rate),
        target_update_frequency=100,
        double_q_model=True,
        huber_loss=0.0,
        exploration=dict(type='linear', unit='timesteps', num_steps=10000, initial_value=1.0, final_value=0.1),
        device=device
    )
