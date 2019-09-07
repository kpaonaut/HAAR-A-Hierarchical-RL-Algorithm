from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
        ]).reshape(-1)
        # return np.concatenate([
        #     self.model.data.qpos.flat,
        #     self.model.data.qvel.flat,
        #     np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        #     self.get_body_xmat("torso").flat,
        #     self.get_body_com("torso"),
        # ]).reshape(-1)

    def step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        yposbefore = self.model.data.qpos.flat[1]
        self.forward_dynamics(action)
        xposafter = self.model.data.qpos[0, 0]
        yposafter = self.model.data.qpos.flat[1]

        right = (xposafter - xposbefore) / self.dt
        up = (-yposafter + yposbefore) / self.dt
        left = (xposbefore - xposafter) / self.dt
        reward = np.array([right, up, left])

        # if self.rew_speed:
        #     direction_com = self.get_body_comvel('torso')
        # else:
        #     direction_com = self.get_body_com('torso')
        # if self.reward_dir:
        #     direction = np.array(self.reward_dir, dtype=float) / np.linalg.norm(self.reward_dir)
        #     forward_reward = np.dot(direction, direction_com)
        # else:
        #     forward_reward = np.linalg.norm(
        #         direction_com[0:-1])  # instead of comvel[0] (does this give jumping reward??)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        # ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05  # this is not in swimmer neither!! And in the GYM env it's 1!!!

        # if self.sparse:  # strip the forward reward, but keep the other costs/rewards!
        #     if np.linalg.norm(self.get_body_com("torso")[0:2]) > np.inf:  # potentially could specify some distance
        #         forward_reward = 1.0
        #     else:
        #         forward_reward = 0.

        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self._state
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.3   and state[2] <= 1.1  # this was 0.2 and 1.0
        done = not notdone
        ob = self.get_current_obs()
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        ori = self.get_ori()
        return Step(ob, reward, done,
                    com=com, ori=ori,
                    contact_cost=contact_cost, survive_reward=survive_reward)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

