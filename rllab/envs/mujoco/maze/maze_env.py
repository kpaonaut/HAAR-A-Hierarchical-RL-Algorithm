import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np

from rllab import spaces
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.mujoco.maze.maze_env_utils import construct_maze
from rllab.envs.mujoco.maze.maze_env_utils import construct_maze_random
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.envs.mujoco.maze.maze_env_utils import ray_segment_intersect, point_distance
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from rllab.misc import logger
import random


class MazeEnv(ProxyEnv, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None
    MAZE_MAKE_CONTACTS = False
    MAZE_STRUCTURE = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    MANUAL_COLLISION = False

    def __init__(
            self,
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            maze_id=0,
            length=1,
            maze_height=0.5,
            maze_size_scaling=4,
            coef_inner_rew=0.,  # a coef of 0 gives no reward to the maze from the wrapped env.
            goal_rew=1000.,  # reward obtained when reaching the goal
            death_reward=0.,
            random_start=False,
            direct_goal=False,
            velocity_field=True,
            visualize_goal=False,
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())
        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span
        self._maze_id = maze_id
        self.length = length
        self.coef_inner_rew = coef_inner_rew
        self.goal_rew = goal_rew
        self.death_reward = death_reward
        self.direct_goal = direct_goal
        self.velocity_field = velocity_field
        self.algo = None # will be added in set_algo!
        self.visualize_goal = visualize_goal

        model_cls = self.__class__.MODEL_CLASS
        # print("model_cls", model_cls)
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        # print("xml_path", xml_path)
        self.tree = tree = ET.parse(xml_path)
        self.worldbody = worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self.init_actual_x = self.init_actual_y = 0
        self.random_start = random_start
        if self.random_start: # in kwargs
            # define: actual coordinates: origin at center of struct (3, 1)
            # define: mujoco coordinates: origin at the first ant
            self.pool, self.p, structure, x_relative, y_relative \
                = construct_maze_random(maze_id=self._maze_id, length=self.length)
            # x, y_relative: x, y index in struct, relative to (3, 1)
            self.MAZE_STRUCTURE = structure
            self.init_actual_x = y_relative * self.MAZE_SIZE_SCALING # map x direction is list y direction!
            self.init_actual_y = x_relative * self.MAZE_SIZE_SCALING
            # print('self init:', self.init_actual_x, self.init_actual_y)
            self.x_r_prev, self.y_r_prev = x_relative + 3, y_relative + 1  # maintain the previous map for later update
            for i, tmp_line in enumerate(structure):
                for j, tmp_member in enumerate(tmp_line):
                    if tmp_member == 'g':
                        self.x_g_prev, self.y_g_prev = i, j
                        break
            # self.x_g_prev, self.y_g_prev = 3, 3 # hard code here! - (1, j) in big maze structure in maze_env_utils.py
            self.x_g, self.y_g = self.x_g_prev, self.y_g_prev
        else:
            self.x_r_prev, self.y_r_prev = 3, 1
            self.x_g_prev, self.y_g_prev = 1, 1
            self.MAZE_STRUCTURE = structure = construct_maze(maze_id=self._maze_id, length=self.length)
        self.tot = 0

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x # this is not the actual init x, just convenient for goal pos computing
        self._init_torso_y = torso_y # torso pos in index coords!
        self.init_torso_x = self._init_torso_x # make visible from outside
        self.init_torso_y = self._init_torso_y

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1"
                    )
                if self.visualize_goal and str(structure[i][j]) == 'g': # visualize goal! uncomment this block when testing!
                    # offset all coordinates so that robot starts at the origin
                    self.goal_element =\
                    ET.SubElement(
                        self.worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - self._init_torso_x,
                                          i * size_scaling - self._init_torso_y,
                                          self.MAZE_HEIGHT / 2 * size_scaling),
                        size="%f %f %f" % (0.2 * size_scaling,
                                           0.2 * size_scaling,
                                           self.MAZE_HEIGHT / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="1.0 0.0 0.0 0.5"
                    )

        self.args = args
        self.kwargs = kwargs

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            # print("geom", geom.attrib)
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        if self.__class__.MAZE_MAKE_CONTACTS:
            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if str(structure[i][j]) == '1':
                        for geom in geoms:
                            ET.SubElement(
                                contact, "pair",
                                geom1=geom.attrib["name"],
                                geom2="block_%d_%d" % (i, j)
                            )

        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)  # here we write a temporal file with the robot specifications. Why not the original one??

        self._goal_range = self._find_goal_range()
        self.goal = np.array([(self._goal_range[0]+self._goal_range[1])/2, (self._goal_range[2]+self._goal_range[3])/2])
        # print ("goal_range", self._goal_range)
        # print("x_y", self.wrapped_env.model.data.qpos.flat[0:2])
        # print("goal", self.goal)
        self._cached_segments = None

        self.gradient_pool = [(1, 0), (0.707, 0.707), (0, 1), (-0.707, 0.707), (-1, 0), (-0.707, -0.707),
                              (0, -1), (0.707, -0.707)] # added gradient pool for train_low_with_v_gradient

        inner_env = model_cls(*args, file_path=file_path, **kwargs)  # file to the robot specifications; model_cls is AntEnv
        ProxyEnv.__init__(self, inner_env)  # here is where the robot env will be initialized

    def set_algo(self, algo): # set the algorithm to the environment
        self.algo = algo

    def get_current_maze_obs(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        robot_x_mujoco, robot_y_mujoco = self.wrapped_env.get_body_com("torso")[:2] # mujoco coords
        # print("mujoco coord:", robot_x_mujoco, robot_y_mujoco)
        robot_x, robot_y = robot_x_mujoco + self._init_torso_x, robot_y_mujoco + self._init_torso_y # index coords
        ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1 or structure[i][j] == 'g':
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)), # index coordinates
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + 1.0 * (2 * ray_idx + 1) / (2 * self._n_bins) * self._sensor_span
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self._sensor_range:
                        wall_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                    if first_seg["distance"] <= self._sensor_range:
                        goal_readings[ray_idx] = (self._sensor_range - first_seg["distance"]) / self._sensor_range
                else:
                    assert False

        # print("wall readings", wall_readings)
        if any(goal_readings) == False:
            x, y = self.wrapped_env.model.data.qpos.flat[0:2]
            print("goal_readings", goal_readings)
            print("x,y", x,y)
            # break

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])
        # return obs
        return obs

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def get_current_obs(self):
        # y, x = self.wrapped_env.model.data.qpos.flat[0:2]
        # if (x <= 1.5 and x > -2.5) or (x > 5.5 and x <= 9.5) or (x > 13.5) or (x <= -6.5):
        #     height = 0
        # elif (x > 1.5 and x <= 5.5) or (x > 9.5 and x <= 13.5) or (x > -6.5 and x <= -2.5):
        #     height = self.MAZE_HEIGHT
        # else:
        #     height = 0

        # x, y = self.wrapped_env.model.data.qpos.flat[0:2]
        # print("get_cuttent_obs", np.concatenate([self.wrapped_env.get_current_obs(),
        #                        self.get_current_maze_obs()
        #                        ]).shape)
        return np.concatenate([self.wrapped_env.get_current_obs(),
                               self.get_current_maze_obs()
                               ])

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
        return self.wrapped_env.model.data.qpos[self.__class__.ORI_IND]

    def reset(self): # randomize starting pos!
        if self.random_start: # in kwargs
            random_i = np.random.choice(len(self.pool), p=self.p) # random i for r
            x_r, y_r = self.pool[random_i]
            x_g, y_g = self.x_g_prev, self.y_g_prev
            random_i = np.random.choice(len(self.pool), p=self.p) # new random i for g
            if self._maze_id == 11: # big maze, goal should also be randomly sampled!
                x_g, y_g = self.pool[random_i]
                while (x_g == x_r) and (y_g == y_r): # shouldn't overlap!
                    random_i = np.random.choice(len(self.pool), p=self.p)
                    x_g, y_g = self.pool[random_i]
                # x_r, y_r = 3, 1 # uncomment for test
                # print('robot:', x_r, y_r)
                # print('goal:', x_g, y_g)
                self.MAZE_STRUCTURE[self.x_g_prev][self.y_g_prev] = 0
                self.MAZE_STRUCTURE[x_g][y_g] = 'g'
                self.x_g_prev, self.y_g_prev = x_g, y_g
                self._goal_range = self._find_goal_range()
                self.goal = np.array(
                    [(self._goal_range[0] + self._goal_range[1]) / 2, (self._goal_range[2] + self._goal_range[3]) / 2])

            # # (3, 1) was the initial choice!
            self.MAZE_STRUCTURE[self.x_r_prev][self.y_r_prev] = 0
            self.x_r_prev, self.y_r_prev = x_r, y_r
            self.MAZE_STRUCTURE[x_r][y_r] = 'r' # update maze
            # # print("x_r, y_r", x_r, y_r)
            # x_relative = x_r - 3  # the x index relative to (0, 0)
            # y_relative = y_r - 1
            # self.init_actual_x = y_relative * self.MAZE_SIZE_SCALING # map x direction is list y direction!
            # self.init_actual_y = x_relative * self.MAZE_SIZE_SCALING
            # # print(self.MAZE_STRUCTURE)
            # # print('self init:', self.init_actual_x, self.init_actual_y)
            x_new_actual = (y_r - 1) * self.MAZE_SIZE_SCALING
            y_new_actual = (x_r - 3) * self.MAZE_SIZE_SCALING
            x_new_mujoco = x_new_actual - self.init_actual_x
            y_new_mujoco = y_new_actual - self.init_actual_y
            qpos = [x_new_mujoco, y_new_mujoco, 0.75, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            qvel = [0.] * 14  # make sure these should be set to 0.?
            qacc = [0.] * 14
            ctrl = [0.] * 14
            init_state = qpos + qvel + qacc + ctrl
            self.wrapped_env.reset(init_state=init_state)  # wapped_env is the inner_env in __init__()
            # in mujoco_env.py, reset(self, init_state=None)
            # print(x_r, y_r)
            if self.visualize_goal: # remove the prev goal and add a new goal
                i, j = x_g, y_g
                size_scaling = self.MAZE_SIZE_SCALING
                # remove the original goal
                try:
                    self.worldbody.remove(self.goal_element)
                except AttributeError:
                    pass
                # offset all coordinates so that robot starts at the origin
                self.goal_element = \
                    ET.SubElement(
                        self.worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - self._init_torso_x,
                                          i * size_scaling - self._init_torso_y,
                                          self.MAZE_HEIGHT / 2 * size_scaling),
                        size="%f %f %f" % (0.2 * size_scaling, # smaller than the block to prevent collision
                                           0.2 * size_scaling,
                                           self.MAZE_HEIGHT / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="1.0 0.0 0.0 0.5"
                    )
                # Note: running the lines below will make the robot position wrong! (because the graph is rebuilt)
                torso = self.tree.find(".//body[@name='torso']")
                geoms = torso.findall(".//geom")
                for geom in geoms:
                    # print("geom", geom.attrib)
                    if 'name' not in geom.attrib:
                        raise Exception("Every geom of the torso must have a name "
                                        "defined")
                _, file_path = tempfile.mkstemp(text=True)
                self.tree.write(
                    file_path)  # here we write a temporal file with the robot specifications. Why not the original one??
                model_cls = self.__class__.MODEL_CLASS
                inner_env = model_cls(*self.args, file_path=file_path,
                                      **self.kwargs)  # file to the robot specifications; model_cls is AntEnv
                ProxyEnv.__init__(self, inner_env)  # here is where the robot env will be initialized


        else:
            self.wrapped_env.reset()
        return self.get_current_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    # space of only the robot observations (they go first in the get current obs) THIS COULD GO IN PROXYENV
    @property
    def robot_observation_space(self):
        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def maze_observation_space(self):
        shp = self.get_current_maze_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def _find_robot(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        i, j = self.x_r_prev, self.y_r_prev
        return j * size_scaling, i * size_scaling
        # for i in range(len(structure)):
        #     for j in range(len(structure[0])):
        #         if structure[i][j] == 'r':
        #             return j * size_scaling, i * size_scaling
        assert False

    def _find_goal_range(self):  # this only finds one goal!
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        i, j = self.x_g_prev, self.y_g_prev
        minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
        maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
        miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
        maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
        return minx, maxx, miny, maxy
        # for i in range(len(structure)):
        #     for j in range(len(structure[0])):
        #         if structure[i][j] == 'g':
        #             minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
        #             maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
        #             miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
        #             maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
        #             return minx, maxx, miny, maxy

    def _is_in_collision(self, pos):
        x, y = pos
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def get_next_maze_obs(self, next_dx, next_dy): # the maze obs with a virtual step! (in order to find the gradient)
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING

        # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-left corner of struc)
        o_xy = np.array(self._find_robot())  # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
        o_ij = (o_xy / size_scaling).astype(int)  # this is the initial position in the grid (check if correct..) [j, i]

        # robot_xy = np.array(self.wrapped_env.get_body_com("torso")[:2])  # the coordinates of this are wrt the init!!
        x_actual, y_actual = self.wrapped_env.get_body_com("torso")[0] + self._init_torso_x - o_xy[0] + next_dx,\
                             self.wrapped_env.get_body_com("torso")[1] + self._init_torso_y - o_xy[1] + next_dy
        robot_xy = np.array([x_actual, y_actual]) # these are the x, y relative to the init pos in this episode

        ori = self.get_ori()  # for Ant this is computed with atan2, which gives [-pi, pi]
        c_ij = o_ij + np.rint(robot_xy / size_scaling) # relative to the grid
        c_xy = (c_ij - o_ij) * size_scaling  # the xy position of the current cell center in init_robot origin
        # print("c_xy", c_xy)
        # print('o_xy', o_xy)
        # print('robot_xy', robot_xy)
        # print('c_ij', c_ij)
        R = int(self._sensor_range // size_scaling)

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (
            self._n_bins - 1) * self._sensor_span  # make the ray in [-pi, pi]
            if ray_ori > math.pi:
                ray_ori -= 2 * math.pi
            elif ray_ori < - math.pi:
                ray_ori += 2 * math.pi
            x_dir, y_dir = 1, 1
            if math.pi / 2. <= ray_ori <= math.pi:
                x_dir = -1
            elif 0 > ray_ori >= - math.pi / 2.:
                y_dir = -1
            elif - math.pi / 2. > ray_ori >= - math.pi:
                x_dir, y_dir = -1, -1

            visited_goal = False
            visited_wall = False
            for r in range(R):
                next_x = c_xy[0] + x_dir * (0.5 + r) * size_scaling  # x of the next vertical segment, in init_rob coord
                next_i = int(c_ij[0] + x_dir * (r + 1))  # this is the i of the cells on the other side of the segment
                delta_y = (next_x - robot_xy[0]) * math.tan(ray_ori)
                y = robot_xy[1] + delta_y  # y of the intersection pt, wrt robot_init origin
                dist = np.sqrt(np.sum(np.square(robot_xy - (next_x, y))))
                # if dist > self._sensor_range:
                #     # print("sensor range is too small")
                #     print("vertical dist", dist)
                if dist <= self._sensor_range and 0 <= next_i < len(structure[0]):
                    j = int(np.rint((y + o_xy[1]) / size_scaling))
                    if 0 <= j < len(structure):
                        if structure[j][next_i] == 1 and not visited_wall:
                            wall_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            # plot_ray(wall_readings[ray_idx], ray_idx)
                            visited_wall = True
                        elif structure[j][next_i] == 'g' and not visited_goal:  # remember to flip the ij when referring to the matrix!!
                            goal_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            # plot_ray(goal_readings[ray_idx], ray_idx, 'g')
                            visited_goal = True
                    else:
                        break
                else:
                    break
            # same for next horizontal segment. If the distance is less (higher intensity), update the goal/wall reading
            visited_goal = False
            visited_wall = False
            for r in range(R):
                next_y = c_xy[1] + y_dir * (0.5 + r) * size_scaling  # y of the next horizontal segment
                next_j = int(
                    c_ij[1] + y_dir * (r + 1))  # this is the i and j of the cells on the other side of the segment
                # first check the intersection with the next horizontal segment:
                delta_x = (next_y - robot_xy[1]) / math.tan(ray_ori)
                x = robot_xy[0] + delta_x
                dist = np.sqrt(np.sum(np.square(robot_xy - (x, next_y))))
                if dist <= self._sensor_range and 0 <= next_j < len(structure):
                    i = int(np.rint((x + o_xy[0]) / size_scaling))
                    if 0 <= i < len(structure[0]):
                        intensity = (self._sensor_range - dist) / self._sensor_range # closeness
                        if structure[next_j][i] == 1 and not visited_wall:
                            if wall_readings[ray_idx] == 0 or intensity > wall_readings[ray_idx]:
                                wall_readings[ray_idx] = intensity
                                # plot_ray(wall_readings[ray_idx], ray_idx)
                            visited_wall = True
                        elif structure[next_j][i] == 'g' and not visited_goal:
                            if goal_readings[ray_idx] == 0 or intensity > goal_readings[ray_idx]:
                                goal_readings[ray_idx] = intensity
                                # plot_ray(goal_readings[ray_idx], ray_idx, 'g')
                            visited_goal = True
                    else:
                        break
                else:
                    break

        obs = np.concatenate([
                wall_readings,
                goal_readings
            ])
        return obs

    def step(self, action):
        if self.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            inner_next_obs, inner_rew, done, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._is_in_collision(new_pos):
                self.wrapped_env.set_xy(old_pos)
                done = False
        else:
            inner_next_obs, inner_rew, done, info = self.wrapped_env.step(action)
        next_obs = self.get_current_obs() # this function is overwritten by the one in sandbox/snn/.../fast_maze_env
        x, y = self.wrapped_env.model.data.qpos.flat[0:2] # this is the x, y relative to the initial pos in mujoco!
        actual_x = x + self.init_actual_x
        actual_y = y + self.init_actual_y
        # dense and right reward
        size_scaling = self.MAZE_SIZE_SCALING

        # print("position", x, y)
        # reward for task 1
        if done:
            reward = self.death_reward
            # print("DEAD!!")
        elif self.velocity_field:
            # the reward defined by velocity field!
            if ((-0.5*size_scaling < actual_x < 1.5*size_scaling) and
                   (-0.5*size_scaling < actual_y < 0.5*size_scaling)):
                # self.goal = np.array([2.0 * size_scaling, 0])
                reward = inner_rew[0]
            elif ((1.5*size_scaling < actual_x  < 2.5*size_scaling) and
                   (-1.5*size_scaling < actual_y < 0.5*size_scaling)):
                # self.goal = np.array([2.0 * size_scaling, 2.0 * size_scaling])
                reward = inner_rew[1]
            elif ((-0.5*size_scaling < actual_x < 2.5*size_scaling) and
                   (-2.5*size_scaling < actual_y < -1.5*size_scaling)):
                reward = inner_rew[2]
            else:
                reward = 0.0
            # print("position", relative_x, relative_y
        elif self.algo is not None and self.algo.train_low_with_v_gradient:
            vel_x = inner_rew[0] # right speed
            vel_y = -inner_rew[1] # down speed (-up speed). down is the positive direction of y!
            value_function = self.algo.baseline
            max_v = -1000.
            gradient_x = gradient_y = 0
            for dx, dy in self.gradient_pool:
                delta_x = dx * self.MAZE_SIZE_SCALING
                delta_y = dy * self.MAZE_SIZE_SCALING
                # add dx, dy to current x, y
                obs = np.concatenate([self.wrapped_env.get_current_obs(), # robot ego observation
                                    self.get_next_maze_obs(delta_x, delta_y)]) # maze obs with a virtual step
                path = {'observations': [obs], 'rewards': [0]} # path contains a single step!
                v = value_function.predict(path)[0]
                if v > max_v:
                    max_v = v
                    gradient_x = dx
                    gradient_y = dy
            reward = np.dot((vel_x, vel_y), (gradient_x, gradient_y)) # inner product of speed
        else:
            reward = 0.0 # no reward for velocity
        # calculate distance to the goal
        if ((-0.5 * size_scaling < actual_x < 1.5 * size_scaling) and
                (-0.5 * size_scaling < actual_y < 0.5 * size_scaling)):
            distance = (1.5*size_scaling - actual_x) + (actual_y + 1.5*size_scaling) + 1.5*size_scaling
        elif ((1.5 * size_scaling < actual_x < 2.5 * size_scaling) and
              (-1.5 * size_scaling < actual_y < 0.5 * size_scaling)):
            distance = (actual_y + 1.5*size_scaling) + actual_x
        elif ((-0.5 * size_scaling < actual_x < 2.5 * size_scaling) and
              (-2.5 * size_scaling < actual_y < -1.5 * size_scaling)):
            distance = actual_x
        else:
            distance = 0.0
        # print("distance", distance)
        info['distance'] = distance
        info['actual_pos'] = actual_x, actual_y
        # info['high_obs'] = next_obs

        minx, maxx, miny, maxy = self._goal_range # these are coordinates in mujoco env relative to the initial rob pos
        # if self.first: # print values while the ant is reset!
        #     print("miny, maxy, minx, maxx", miny, maxy, minx, maxx, "actualx, actualy", actual_x, actual_y)
        #     self.first = False
        if  (miny <= y <= maxy) and (minx <= x <= maxx):
            done = True
            # print("SUCCESS!")
            reward += self.goal_rew # still add 100, to give a fair comparison
            info['inner_rew'] = 1.  # we keep here the original one, so that the AverageReturn is directly the freq of success
        else:
            info['inner_rew'] = 0.
        info['outer_rew'] = reward
        # print("reward", reward)
        return Step(next_obs, reward, done, **info)

    def action_from_key(self, key):
        return self.wrapped_env.action_from_key(key)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        # we call here any logging related to the maze, strip the maze obs and call log_diag with the stripped paths
        # we need to log the purely gather reward!!
        with logger.tabular_prefix('Maze_'):
            gather_undiscounted_returns = [sum(path['env_infos']['outer_rew']) for path in paths]
            logger.record_tabular_misc_stat('Return', gather_undiscounted_returns, placement='front')
        stripped_paths = []
        for path in paths:
            stripped_path = {}
            for k, v in path.items():
                # print("k", k)
                stripped_path[k] = v
            # for k, v in path["agent_infos"].items():
            #     print("k", k)
            # print("latents", stripped_path["agent_infos"]["latents"])
            # print("latents", stripped_path["agent_infos"]["latents"].shape)
            # print("shape_len", len(stripped_path['observations'].shape))
            # print("after_con", np.concatenate(stripped_path['observations']).shape)

            if len(stripped_path['observations'].shape) == 1:
                stripped_path['observations'] = np.concatenate(stripped_path['observations'])

            stripped_path['observations'] = \
                stripped_path['observations'][:, :self.wrapped_env.observation_space.flat_dim]
            #  this breaks if the obs of the robot are d>1 dimensional (not a vector)
            stripped_paths.append(stripped_path)
        with logger.tabular_prefix('wrapped_'):
            wrapped_undiscounted_return = np.mean([np.sum(path['env_infos']['inner_rew']) for path in paths])
            # for _ in range(10):
            #     print('OK!')
            # print(wrapped_undiscounted_return)
            # print([np.sum(path['env_infos']['inner_rew']) for path in paths])
            logger.record_tabular('SuccessRate', wrapped_undiscounted_return)
            self.wrapped_env.log_diagnostics(stripped_paths, *args, **kwargs)

    def get_reward_fn(self):
        return -np.sum(np.square(self.wrapped_env.model.data.qpos.flat[0:2] - self.goal)) ** 0.5