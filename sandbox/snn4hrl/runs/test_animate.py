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
import configs.config_for_transfer as par

stub(globals())

# exp setup --------------------------------------------------------
mode = "local"
n_parallel = 1

# exp_dir = '/home/lsy/Desktop/rllab/data/local/Ant-snn/'
# for dir in os.listdir(exp_dir):
#     if 'Figure' not in dir and os.path.isfile(os.path.join(exp_dir, dir, 'params.pkl')):
#         pkl_path = os.path.join(exp_dir, dir, 'params.pkl')

pkl_path = '/home/wr1/rllab/data/local/four-room/40TrueTrainLowVSplit_Falseanneal_12parallel_2000000LowStepNum_10000MaxLowStep_100timestepagg_2019_05_18_15_33_52/params.pkl'
#pkl_path = '/home/wr1/rllab/data/local/par-tuning/1parallel_1trainhighevery_0.99discounthigh_2019_04_18_11_28_57/params.pkl'
data = joblib.load(pkl_path)
algo = data["algo"]
if "time_steps_agg" in data:
    algo.env.time_steps_agg = data["time_steps_agg"]

warm_params = data["policy"].get_params_internal()
algo.policy.set_params_snn(warm_params) # looks like these params are saved in the pkl file and will not be overridden
warm_params_low = algo.low_policy.get_params_internal()

params_env_low_policy = algo.env.low_policy.get_params_internal()
algo.env.random_start = True
algo.env.wrapped_env.wrapped_env.visualize_goal = True # visualize the goal!

algo.low_policy.set_params_snn(warm_params_low)
algo.env.low_policy.set_params_snn(warm_params_low) # must use env.low_policy! updating algo.low_policy is useless

algo.env.animate = True

algo.batch_size = 2e3/50

s = 10
exp_prefix = 'test'
exp_name = 'test'

run_experiment_lite(
    stub_method_call=algo.train(),
    mode=par.mode,
    use_cloudpickle=False,
    pre_commands=['pip install --upgrade pip',
                  'pip install --upgrade theano',
                  ],
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    seed=s,
    # Save to data/local/exp_prefix/exp_name/
    exp_prefix=exp_prefix,
    exp_name=exp_name,
    use_gpu=False,
)