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

def get_dueling_dqn_agent(model, environment, device='CPU', **kwargs):
    return Agent.create(
        agent='dueling_dqn',
        environment=environment,
        memory=10000,
        batch_size=64,
        network=model,
        device=device,
        update_frequency=1,
        target_update_weight=0.05,
        start_updating=1000,
        **kwargs
    )
