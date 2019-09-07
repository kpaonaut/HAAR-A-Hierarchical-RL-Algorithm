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
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.snn4hrl.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv

stub(globals())

# exp setup --------------------------------------------------------
mode = "local"
n_parallel = 16
train_low = False
train_high = True
maze_id = 11
fence = False
n_itr = 100
death_reward = -10.0
sensor_range = 40.0
low_step_num = 6e5 # big maze: 6e5
max_low_step = 3e3
success_reward = 1000.
exp_prefix_set = 'four-room'

animate = False
direct_goal = False
random_start = True
velocity_field = False
train_high_every = 1
max_path_length = 3e3

for maze_size_scaling in [4]:

    # for time_step_agg in [50, 100, 10]:
    env = normalize(AntMazeEnv(maze_id=maze_id, death_reward=death_reward, sensor_range=sensor_range,
                                     sensor_span=math.pi * 2, ego_obs=True, fence=fence,
                                     goal_rew=success_reward, random_start=random_start,
                                    direct_goal=direct_goal, velocity_field=velocity_field,
                                     ))


    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # latent_dim=latent_dim,
        # latent_name='categorical',
        # bilinear_integration=True,  # concatenate also the outer product
        hidden_sizes=(64, 64),
        min_std=1e-6,
    )
    # print("env_hier", env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    batch_size = low_step_num
    # batch_size = 20


    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        baselinename='linear',
        self_normalize=True,
        log_deterministic=True,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=0.99,
        discount_high=0.99,
        step_size=0.01,
        train_low=train_low,
        train_high=train_high,
        train_high_every=train_high_every,
        train_low_with_penalty=False,
        train_low_with_v_split=False,
        train_low_with_v_gradient=False,
        time_step_agg_anneal=False,
        total_low_step=low_step_num,
        episode_max_low_step=max_low_step,
        low_level_entropy_penalty=0.,
        itr_delay=0,
    )

    for s in [0]:  # range(10, 110, 10):  # [10, 20, 30, 40, 50]:
        exp_prefix = exp_prefix_set
        # exp_prefix = "test"
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        exp_name = 'TRPO_scratch__{}batch_{}length_{}id_{}_parallel{}'.format(
                                                        # time_step_agg,
                                                        int(batch_size),
                                                        int(max_path_length), maze_id,
                                                        timestamp, n_parallel)

        run_experiment_lite(
            stub_method_call=algo.train(),
            mode=mode,
            use_cloudpickle=False,
            pre_commands=['pip install --upgrade pip',
                          'pip install --upgrade theano',
                          ],
            # Number of parallel workers for sampling
            n_parallel=n_parallel,

            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=s,
            # Save to data/local/exp_prefix/exp_name/
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            use_gpu=False,
        )
