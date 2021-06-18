import numpy as np
from gym import utils

from gather_env.envs.gather_env import GatherEnv
from gather_env.envs.mymujoco import MyMujocoEnv


class MySwimmerEnv(MyMujocoEnv, utils.EzPickle):
    FILE = "swimmer_gather.xml"

    def __init__(self, xml_path, *args, **kwargs):
        MyMujocoEnv.__init__(self, xml_path, 50)  # TODO: check frame skip from file
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self.get_current_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def get_current_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self.get_current_obs()


class SwimmerGatherEnv(GatherEnv):
    MODEL_CLASS = MySwimmerEnv
    ORI_IND = 2

