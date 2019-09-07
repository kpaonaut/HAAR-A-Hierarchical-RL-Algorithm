from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs


class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-4,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            # self.get_body_com("torso").flat,
        ]).reshape(-1)

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        # self.forward_dynamics(action)
        # next_obs = self.get_current_obs()
        # lb, ub = self.action_bounds
        # scaling = (ub - lb) * 0.5
        # ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
        #     np.square(action / scaling))
        # forward_reward = self.get_body_comvel("torso")[0]
        # reward = forward_reward - ctrl_cost
        # done = False
        # return Step(next_obs, reward, done)

        ctrl_cost_coeff = 0.0001
        xposbefore = self.model.data.qpos[0, 0]
        yposbefore = self.model.data.qpos.flat[1]
        self.forward_dynamics(action, 4)
        xposafter = self.model.data.qpos[0, 0]
        yposafter = self.model.data.qpos.flat[1]

        right = (xposafter - xposbefore) / self.dt
        up = (yposafter - yposbefore) / self.dt
        left = (xposbefore - xposafter) / self.dt
        down = (yposbefore - yposafter) / self.dt
        reward = np.array([right, up, left, down])

        # reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(action).sum()
        reward = reward + reward_ctrl
        #reward = (reward[0]**2 + reward[1]**2 + reward[2]**2 + reward[3]**2)**0.5 # Rui: change reward dimension
        ob = self.get_current_obs()
        return ob, reward, False, dict(reward_fwd=reward, reward_ctrl=reward_ctrl)

    @overrides
    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)