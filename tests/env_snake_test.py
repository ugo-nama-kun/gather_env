import numpy as np

from gather_env.envs.snake_gather_env import SnakeGatherEnv


class TestEnv:
    def test_instance(self):
        env = SnakeGatherEnv()

    def test_reset_env(self):
        env = SnakeGatherEnv()
        env.reset()

    def test_dim(self):
        env = SnakeGatherEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 12 + 10 + 10  # 17 + 10 + 10 if non-ego-centric observation
        assert len(env.action_space.high) == 4
        assert len(obs) == 12 + 10 + 10  # 17 + 10 + 10 if non-ego-centric observation
        assert len(env.action_space.sample()) == 4

    def test_terminal(self):
        env = SnakeGatherEnv(max_episode_steps=100)
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
        env = SnakeGatherEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_render_env(self):
        env = SnakeGatherEnv()
        for n in range(5):
            env.reset()
            for i in range(100):
                env.step(env.action_space.sample())
                env.render()
        env.close()

    def test_reset(self):
        env = SnakeGatherEnv()
        env.reset()
        initial_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        for i in range(1000):
            env.step(env.action_space.sample())

        env.reset()
        reset_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        np.testing.assert_allclose(actual=reset_robot_pos,
                                   desired=initial_robot_pos,
                                   atol=0.3)