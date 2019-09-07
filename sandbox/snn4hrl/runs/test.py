import math
import datetime
import dateutil.tz
import joblib
import pickle

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

pkl_path = '/home/wr1/rllab/data/local/params.pkl'
# pkl_path = '/home/wr1/rllab/data/local/four-room/40TrueTrainLowVSplit_Falseanneal_12parallel_2000000LowStepNum_10000MaxLowStep_100timestepagg_2019_05_18_15_33_52/params.pkl'

data = joblib.load(pkl_path)

algo = data["algo"]
if "time_steps_agg" in data:
    algo.env.time_steps_agg = data["time_steps_agg"]

warm_params = data["policy"].get_params_internal()
warm_params_low = algo.low_policy.get_params_internal()

with open('/home/wr1/rllab/data/local/transfer/warm_params.pkl', 'wb') as output:
    pickle.dump(warm_params, output)
    pickle.dump(warm_params, output)