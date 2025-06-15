from tensorforce import Agent

def get_ppo_agent(environment, network, device='CPU', **kwargs):
    return Agent.create(
        agent='ppo',
        environment=environment,
        network=network,
        batch_size=10,
        learning_rate=1e-4,
        update_frequency=2,
        max_episode_timesteps=1000,
        device=device,
        **kwargs
    )


def get_dueling_dqn_agent(environment, device='CPU', **kwargs):
    network_spec = [
        dict(type='dense', size=64, activation='relu'),
        dict(type='dense', size=64, activation='relu')
    ]

    return Agent.create(
        agent='dueling_dqn',
        environment=environment,
        memory=10000,
        batch_size=64,
        update_frequency=1,
        target_update_weight=0.05,
        start_updating=1000,
        network=network_spec,
        device=device,
        **kwargs
    )


