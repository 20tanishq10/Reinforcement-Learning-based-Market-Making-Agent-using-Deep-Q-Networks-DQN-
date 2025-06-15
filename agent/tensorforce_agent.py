from tensorforce import Agent

def get_ppo_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    return Agent.create(
        agent='ppo',
        environment=environment,
        max_episode_timesteps=max_episode_timesteps,
        config=dict(
            network=model,
            batch_size=10,
            update_frequency=10,
            learning_rate=learning_rate,
            device=device
        )
    )

def get_dueling_dqn_agent(model, environment, max_episode_timesteps, device='cpu', learning_rate=1e-4, horizon=1):
    return Agent.create(
        agent='dqn',
        environment=environment,
        max_episode_timesteps=max_episode_timesteps,
        config=dict(
            network=model,
            memory=dict(type='replay', capacity=10000),
            update=dict(unit='timesteps', batch_size=32),
            optimizer=dict(type='adam', learning_rate=learning_rate),
            target_update_frequency=100,
            exploration=0.1,
            device=device
        )
    )
