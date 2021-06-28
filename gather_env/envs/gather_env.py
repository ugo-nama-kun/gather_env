import math
import os
import tempfile
import xml.etree.ElementTree as ET
import inspect

import numpy as np
import glfw

from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv, DEFAULT_SIZE
from gym import utils
from mujoco_py.generated import const


APPLE = 0
BOMB = 1
BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}


class GatherEnv(MujocoEnv, utils.EzPickle):
    MODEL_CLASS = None
    ORI_IND = None
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 ego_obs=True,
                 n_apples=8,
                 n_bombs=8,
                 activity_range=6.,
                 robot_object_spacing=2.,
                 catch_range=1.,
                 n_bins=10,
                 sensor_range=6.,
                 sensor_span=2 * math.pi,  # actually this parameter is used in the previous study by substituting the value
                 coef_inner_rew=0.,  # do not use inner reward. as is in hippo code
                 dying_cost=-10,
                 max_episode_steps=math.inf,
                 *args, **kwargs):
        """

        :param int n_apples:  Number of apples in each episode
        :param int n_bombs: Number of bombs in each episode
        :param float activity_range: he span for generating objects (x, y in [-range, range])
        :param float robot_object_spacing: Number of objects in each episode
        :param float catch_range: Minimum distance range to catch an object
        :param int n_bins: Number of objects in each episode
        :param float sensor_range: Maximum sensor range (how far it can go)
        :param float sensor_span: Maximum sensor span (how wide it can span), in radians
        :param coef_inner_rew:
        :param dying_cost:
        :param args:
        :param kwargs:
        """
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.objects = []
        self.viewer = None

        utils.EzPickle.__init__(**locals())

        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)

        tree = ET.parse(MODEL_DIR)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)

        # build mujoco
        self.wrapped_env = model_cls(file_path, **kwargs)

        # optimization, caching obs spaces
        ub = BIG * np.ones(self.get_current_obs().shape)
        self.obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(self.get_current_robot_obs().shape)
        self.robot_obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(np.concatenate(self.get_readings()).shape)
        self.maze_obs_space = spaces.Box(ub * -1, ub)

        self._action_space = None

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset(self):
        self.wrapped_env.reset()
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

        return self.get_current_obs()

    def step(self, action):
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rew'] = inner_rew
        info['outer_rew'] = 0
        if done:
            return self.get_current_obs(), self.dying_cost, done, info  # give a -10 rew if the robot dies
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                    info['outer_rew'] = 1
                else:
                    reward = reward - 1
                    info['outer_rew'] = -1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return self.get_current_obs(), reward, done, info

    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb;
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        apple_readings, bomb_readings = self.get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings])

    @property
    def observation_space(self):
        return self.obs_space

    # space of only the robot observations (they go first in the get current obs)
    @property
    def robot_observation_space(self):
        return self.robot_obs_space

    @property
    def maze_observation_space(self):
        return self.maze_obs_space

    @property
    def action_space(self):
        if self._action_space is None:
            return self.wrapped_env.action_space
        else:
            return self._action_space

    @action_space.setter
    def action_space(self, new_space):
        self._action_space = new_space

    @property
    def action_bounds(self):
        return self.wrapped_env.action_bounds

    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        obj = self.wrapped_env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.sim.data.qpos[self.__class__.ORI_IND]

    def close(self):
        if self.wrapped_env.viewer:
            glfw.destroy_window(self.wrapped_env.viewer.window)
            self.viewer = None

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        self.wrapped_env.render(mode, width, height, camera_id, camera_name)
        for obj in self.objects:
            ox, oy, typ = obj
            if typ == BOMB:
                rgba = (1, 0, 0, 1)
            else:
                rgba = (0, 1, 0, 1)
            self.wrapped_env.viewer.add_marker(pos=np.array([ox, oy, 0.5]),
                                               label=" ",
                                               type=const.GEOM_SPHERE,
                                               size=(0.5, 0.5, 0.5),
                                               rgba=rgba)
