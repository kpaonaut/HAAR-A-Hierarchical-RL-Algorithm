from rllab.algos.trpo import TRPO
from rllab.algos.tnpg import TNPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from contrib.alexbeloi.is_sampler import ISSampler
from rllab.misc.instrument import stub, run_experiment_lite

"""
Example using VPG with ISSampler, iterations alternate between live and
importance sampled iterations.
"""

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)
# baseline = GaussianMLPBaseline(env_spec=env.spec)

optimizer_args = dict(
    # debug_nan=True,
    # reg_coeff=0.1,
    # cg_iters=2
)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=200,
    discount=0.99,
    step_size=0.01,
    sampler_cls=ISSampler,
    sampler_args=dict(n_backtrack=1),
    optimizer_args=optimizer_args
)
# algo.train()
run_experiment_lite(
                stub_method_call=algo.train(),
                mode="local",
                use_cloudpickle=False,
                pre_commands=['pip install --upgrade pip',
                              'pip install --upgrade theano',
                              ],
                # Number of parallel workers for sampling
                n_parallel=1,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=0,
                # Save to data/local/exp_prefix/exp_name/
                exp_prefix="cart_pole",
                exp_name="linear",
                use_gpu=False,
            )
