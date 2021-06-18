import numpy as np

from gather_env.envs.ant_gather_env import AntGatherEnv


class TestEnv:
    def test_instance(self):
        env = AntGatherEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = AntGatherEnv()
        env.reset()

    def test_instance_not_ego_obs(self):
        env = AntGatherEnv(
            ego_obs=False,
            no_contact=False,
            sparse=False
        )
        env.reset()

    def test_instance_no_contact(self):
        env = AntGatherEnv(
            ego_obs=True,
            no_contact=True,
            sparse=False
        )
        env.reset()

    def test_dim(self):
        env = AntGatherEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 10
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 10 + 10
        assert len(env.action_space.sample()) == 8

    def test_run_env(self):
        env = AntGatherEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_render_env(self):
        env = AntGatherEnv()
        for n in range(5):
            env.reset()
            for i in range(10):
                env.step(env.action_space.sample())
                env.render()
        env.close()

    def test_reset(self):
        env = AntGatherEnv()
        env.reset()
        initial_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        for i in range(1000):
            env.step(env.action_space.sample())

        env.reset()
        reset_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        np.testing.assert_allclose(actual=reset_robot_pos,
                                   desired=initial_robot_pos,
                                   atol=0.3)
