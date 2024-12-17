import gymnasium


gymnasium.register(
    id='GraspEnv_v1',
    entry_point='graspenvs.graspenv-v1:GraspEnv_v1'
)

gymnasium.register(
    id='GraspEnv_v2',
    entry_point='graspenvs.graspenv-v2:GraspEnv_v2'
)

gymnasium.register(
    id='GraspEnv_v3',
    entry_point='graspenvs.graspenv-v3:GraspEnv_v3'
)
