from sandbox.snn4hrl.envs.mujoco.maze.fast_maze_env import FastMazeEnv
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv
import math
from contextlib import contextmanager

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.mujoco_env import BIG
from rllab.misc.overrides import overrides


class AntMazeEnv(FastMazeEnv):

    MODEL_CLASS = AntEnv
    # MODEL_CLASS.FILE = 'ant1.xml'
    ORI_IND = 3  # the ori of Ant requires quaternion conversion and is implemented in AntEnv

    MAZE_HEIGHT = 3
    MAZE_SIZE_SCALING = 3.0

    # def __init__(
    #         self,
    #         *args,
    #         **kwargs):
    #     MODEL_CLASS = AntEnv
    #     MODEL_CLASS.FILE = 'ant1.xml'
    def __init__(
            self,
            fence=False,
            *args,
            **kwargs):
        MODEL_CLASS = AntEnv
        if fence:
            MODEL_CLASS.FILE = 'ant1.xml'
        else:
            MODEL_CLASS.FILE = 'ant.xml'

        Serializable.quick_init(self, locals())
        MazeEnv.__init__(self, *args, **kwargs)
        self._blank_maze = False
        # add goal obs
        self.blank_maze_obs = np.concatenate([np.zeros(self._n_bins), np.zeros(self._n_bins)])
        # self.blank_maze_obs = np.zeros(self._n_bins)

        # The following caches the spaces so they are not re-instantiated every time
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self._observation_space = spaces.Box(ub * -1, ub)

        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        self._robot_observation_space = spaces.Box(ub * -1, ub)

        shp = self.get_current_maze_obs().shape
        ub = BIG * np.ones(shp)
        self._maze_observation_space = spaces.Box(ub * -1, ub)

