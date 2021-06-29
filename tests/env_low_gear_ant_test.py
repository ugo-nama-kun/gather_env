import numpy as np

from gather_env.envs.low_gear_ant_gather_env import LowGearAntGatherEnv


class TestEnv:
    def test_instance(self):
        env = LowGearAntGatherEnv()

    def test_reset_env(self):
        env = LowGearAntGatherEnv()
        env.reset()

    def test_dim(self):
        env = LowGearAntGatherEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 10
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 10 + 10
        assert len(env.action_space.sample()) == 8

    def test_terminal(self):
        env = LowGearAntGatherEnv(max_episode_steps=100)
        env.reset()
        for i in range(100):
            _, _, done, _ = env.step(env.action_space.sample())

            print(i, done, env._max_episode_steps)
            if i == 99:
                assert done is True
            else:
                assert done is False
        env.close()

    def test_run_env(self):
        env = LowGearAntGatherEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_render_env(self):
        env = LowGearAntGatherEnv()
        for n in range(5):
            env.reset()
            for i in range(10):
                env.step(env.action_space.sample())
                env.render()
        env.close()

    def test_reset(self):
        env = LowGearAntGatherEnv()
        env.reset()
        initial_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        for i in range(1000):
            env.step(env.action_space.sample())

        env.reset()
        reset_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        np.testing.assert_allclose(actual=reset_robot_pos,
                                   desired=initial_robot_pos,
                                   atol=0.3)