from gym.envs.registration import register

register(
    id='MountainCarSparse-v0',
    entry_point='R3L.rl.custom_envs:MountainCarEnv',
    max_episode_steps=200,
)

register(
    id='PendulumSparse-v0',
    entry_point='R3L.rl.custom_envs:PendulumEnv',
    max_episode_steps=100,
)

register(
    id='AcrobotContinuous-v0',
    entry_point='R3L.rl.custom_envs:AcrobotEnv',
    max_episode_steps=500,
)

register(
    id='CartpoleSwingup-v0',
    entry_point='R3L.rl.custom_envs:CartpoleSwingupEnv',
    max_episode_steps=500,
)

register(
    id='ReacherSparse-v0',
    entry_point='R3L.rl.custom_envs:ReacherEnv',
    max_episode_steps=50,
)
