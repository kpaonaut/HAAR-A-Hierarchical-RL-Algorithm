from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.snn4hrl.algos.trpo_snn import TRPO_snn
from sandbox.snn4hrl.bonus_evaluators.grid_bonus_evaluator import GridBonusEvaluator
from sandbox.snn4hrl.envs.mujoco.ant_env_backup import AntEnv
from sandbox.snn4hrl.policies.snn_mlp_policy import GaussianMLPPolicy_snn
from sandbox.snn4hrl.regressors.latent_regressor import Latent_regressor
import joblib
import os
from rllab import config


pkl_path = '/home/lsy/Projects/rllab/data/local/ini-par/Truetrain_low_v_split_Trueanneal_1parallel_1trainhighevery_0.95discounthigh_linearbaseline_2019_04_23_19_03_36/params.pkl'
data = joblib.load(pkl_path)
algo_new = data["algo"]
warm_params_low = algo_new.low_policy.get_params_internal()

pkl_path_snn = '/home/lsy/Desktop/rllab/data/local/Ant-snn1000/Ant-snn_10MI_5grid_6latCat_bil_0040/params.pkl'
pkl_path_snn = os.path.join(config.PROJECT_PATH, pkl_path_snn)
data_snn = joblib.load(pkl_path_snn)
data_snn["iter"] = 100
# print("data_snn", data_snn)
algo_snn = data_snn["algo"]


algo_snn.policy.set_params_snn(warm_params_low)


algo_snn.current_itr = 998
algo_snn.test("/home/lsy/test1/")


