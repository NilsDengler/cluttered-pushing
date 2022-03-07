from gym.envs.registration import register

register(
    id='pushing-v0',
    entry_point='push_gym.tasks.pushing:Pushing',
    kwargs={'render': False, 'use_egl': True}
)
register(
    id='pushingGUI-v0',
    entry_point='push_gym.tasks.pushing:Pushing',
    kwargs={'render': True, 'use_egl': False, "hz": 240}
)
