import numpy as np
from gym import utils

from gather_env.envs.gather_env import GatherEnv
from gather_env.envs.mymujoco import MyMujocoEnv


class MySnakeEnv(MyMujocoEnv, utils.EzPickle):
    FILE = "snake_gather.xml"
    ORI_IND = 2

    def __init__(self,
                 xml_path,
                 ctrl_cost_coeff=1e-2,
                 ego_obs=True,
                 sparse_rew=False,
                 *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.ego_obs = ego_obs
        self.sparse_rew = sparse_rew
        MyMujocoEnv.__init__(self, xml_path, 50)
        utils.EzPickle.__init__(self)

    def get_current_obs(self):
        if self.ego_obs:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
            ]).reshape(-1)
        else:
            # Comment from Yossy:
            # 17 dim observation is descried in HiPPO paper. However, ego-centric observation (less than 17 dim) is used in the code (^^;)
            # https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/runs/pg_test.py#L220
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.get_body_com("torso").flat,
            ]).reshape(-1)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        next_obs = self.get_current_obs()
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(a / scaling))
        forward_reward = np.linalg.norm(self.get_body_comvel("torso"))  # swimmer has no problem of jumping reward
        reward = forward_reward - ctrl_cost
        done = False
        # Written in original Snake.
        # https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/envs/mujoco/snake_env.py#L12
        if self.sparse_rew:
            if abs(self.get_body_com("torso")[0]) > 100.0:
                reward = 1.0
                done = True
            else:
                reward = 0.
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        ori = self.get_ori()
        info = dict(reward_fwd=forward_reward, reward_ctrl=ctrl_cost, com=com, ori=ori)
        return next_obs, reward, done, info

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self.get_current_obs()

    def get_ori(self):
        return self.sim.data.qpos[self.__class__.ORI_IND]

    def get_body_comvel(self, body_name):
        # Imported from https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/rllab/envs/mujoco/mujoco_env.py#L236
        # idx = self.sim.body_names.index(body_name)
        # return self.sim.body_comvels[idx]
        # TODO: check get_body_xvelp == sim.body_comvels
        return self.sim.data.get_body_xvelp(body_name)


class SnakeGatherEnv(GatherEnv):
    MODEL_CLASS = MySnakeEnv
    ORI_IND = MySnakeEnv.ORI_IND

