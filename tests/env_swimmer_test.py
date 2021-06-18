import numpy as np

from gather_env.envs.swimmer_gather_env import SwimmerGatherEnv


class TestEnv:
    def test_instance(self):
        env = SwimmerGatherEnv()

    def test_reset_env(self):
        env = SwimmerGatherEnv()
        env.reset()

    def test_dim(self):
        env = SwimmerGatherEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 8 + 10 + 10
        assert len(env.action_space.high) == 2
        assert len(obs) == 8 + 10 + 10
        assert len(env.action_space.sample()) == 2

    def test_run_env(self):
        env = SwimmerGatherEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_render_env(self):
        env = SwimmerGatherEnv()
        for n in range(5):
            env.reset()
            for i in range(100):
                env.step(env.action_space.sample())
                env.render()
        env.close()

    def test_reset(self):
        env = SwimmerGatherEnv()
        env.reset()
        initial_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        for i in range(1000):
            env.step(env.action_space.sample())

        env.reset()
        reset_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        np.testing.assert_allclose(actual=reset_robot_pos,
                                   desired=initial_robot_pos,
                                   atol=0.3)