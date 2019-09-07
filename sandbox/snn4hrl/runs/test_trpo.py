import math
import datetime
import dateutil.tz
import joblib

from rllab import config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.config_personal import *
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from sandbox.snn4hrl.algos.trpo_snn import TRPO_snn
from sandbox.snn4hrl.envs.hierarchized_snn_env import hierarchize_snn
from sandbox.snn4hrl.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from sandbox.snn4hrl.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv

stub(globals())

# exp setup --------------------------------------------------------
mode = "local"
n_parallel = 1


# pkl_path = '/home/lsy/Desktop/rllab/data/local/fence-False-done-10-sen20-ep1000-new4/50agg_1000batch_20length_8id_par1_low_True_2019_04_15_11_55_30/params.pkl'
pkl_path = '/home/lsy/Desktop/rllab/data/local/fence-False-done-10-sen20-ep1000-new4/50000batch_1000length_8id_2019_04_15_22_12_16_par16/params.pkl'
data = joblib.load(pkl_path)


algo = data["algo"]

# warm_params = data["policy"].get_params_internal()
# algo.policy.set_params_snn(warm_params) # looks like these params are saved in the pkl file and will not be overridden
# warm_params_low = algo.low_policy.get_params_internal()
#
# params_env_low_policy = algo.env.low_policy.get_params_internal()
#
# algo.low_policy.set_params_snn(warm_params_low)
# algo.env.low_policy.set_params_snn(warm_params_low) # must use env.low_policy! updating algo.low_policy is useless

# algo.batch_size = 5e1
print("sensor range", algo.env)
# env = normalize(AntMazeEnv(maze_id=8, death_reward=-10, sensor_range=2000,
#                                      sensor_span=math.pi * 2, ego_obs=True, fence=False,
#                                      goal_rew=1000, random_start=False,
#                                      ))
# algo.env = env
algo.test("/home/lsy/test/new/")
