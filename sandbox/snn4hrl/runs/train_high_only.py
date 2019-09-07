"""
SwimmerGather: find good size to compare agains baseline
"""

# imports -----------------------------------------------------
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
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
import configs.config_for_train_high_only as par # import parameters

stub(globals())



print("hier for : ", par.pkl_path)

for maze_size_scaling in [4]:

    for time_step_agg in [100]:
        inner_env = normalize(AntMazeEnv(maze_id=par.maze_id, death_reward=par.death_reward, sensor_range=par.sensor_range,
                                         sensor_span=math.pi * 2, ego_obs=True, fence=par.fence,
                                         goal_rew=par.success_reward, random_start=par.random_start,
                                         direct_goal=par.direct_goal, velocity_field=par.velocity_field,
                                         ))
        env = hierarchize_snn(inner_env, time_steps_agg=time_step_agg, pkl_path=par.pkl_path,
                              animate=par.animate,
                              )

        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
        )
        print("env_hier", env.spec)
        if par.baseline_name == 'linear':
            baseline = LinearFeatureBaseline(env_spec=env.spec)
        elif par.baseline_name == 'mlp':
            baseline = GaussianMLPBaseline(env_spec=env.spec)

        batch_size = int(par.low_step_num / time_step_agg)
        max_path_length = int(par.max_low_step / time_step_agg)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            # baselinename=par.baseline_name,
            self_normalize=True,
            log_deterministic=True,
            batch_size=batch_size,
            whole_paths=True,
            max_path_length=max_path_length,
            n_itr=par.n_itr,
            discount=0.99, # not used
            discount_low=0.99,
            discount_high=par.discount_high, # 0.99 prev, Rui wants to change it to 0.8 (for time_step_agg == 100)
            train_high_every=par.train_high_every, # 1 prev, Rui wants to change it to 10 (for time_step_agg == 100)
            step_size=0.01,
            train_low=par.train_low,
            train_high=par.train_high,
            train_low_with_penalty=par.train_low_with_penalty,
            train_low_with_v_split=par.train_low_with_v_split,
            time_step_agg_anneal=par.time_step_agg_anneal,
            total_low_step=par.low_step_num,
            episode_max_low_step=par.max_low_step,
            low_level_entropy_penalty=par.low_level_entropy_penalty,
            itr_delay=par.itr_delay,
        )

        for s in [20, 40, 60, 80]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
            exp_prefix = par.exp_prefix_set
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

            exp_name = 'SNN_withLowPenalty_{}anneal_{}parallel_{}LowStepNum_{}MaxLowStep_{}HighDiscount_{}'.format(
                par.time_step_agg_anneal,
                par.n_parallel,
                int(par.low_step_num),
                int(par.max_low_step),
                par.discount_high,
                timestamp)

            run_experiment_lite(
                stub_method_call=algo.train(),
                mode=par.mode,
                use_cloudpickle=False,
                pre_commands=['pip install --upgrade pip',
                              'pip install --upgrade theano',
                              ],
                # Number of parallel workers for sampling
                n_parallel=par.n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=s,
                # Save to data/local/exp_prefix/exp_name/
                exp_prefix=exp_prefix,
                exp_name=exp_name,
                use_gpu=False,
            )
