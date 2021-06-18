import numpy as np
from gym import utils

from gather_env.envs.gather_env import GatherEnv
from gather_env.envs.ant_gather_env import MyAntEnv


class MyLowGearAntEnv(MyAntEnv, utils.EzPickle):
    # TODO: MaKe low-gear version as an option in AntGather
    FILE = "low_gear_ratio_ant_gather.xml"


class LowGearAntGatherEnv(GatherEnv):
    MODEL_CLASS = MyLowGearAntEnv
    ORI_IND = 3
