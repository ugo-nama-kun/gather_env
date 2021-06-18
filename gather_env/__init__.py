from gym.envs.registration import register

register(
    id='AntGather-v0',
    entry_point='gather_env.envs:AntGatherEnv',
)

register(
    id='LowGearAntGather-v0',
    entry_point='gather_env.envs:LowGearAntGatherEnv',
)

register(
    id='SnakeGather-v0',
    entry_point='gather_env.envs:SnakeGatherEnv',
)

register(
    id='SwimmerGather-v0',
    entry_point='gather_env.envs:SwimmerGatherEnv',
)
