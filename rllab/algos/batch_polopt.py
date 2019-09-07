import joblib
from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy
from rllab.misc import ext
import numpy as np
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv
from sandbox.snn4hrl.sampler.low_sampler import LowSampler
from sandbox.snn4hrl.sampler.utils import process_path
#import psutil
import os
import pickle
from rllab.envs.normalized_env import normalize
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
import gc

opt_algorithm = 0 # use this as the global variable

class BatchSampler(BaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        try:
            parallel_sampler.populate_task(self.algo.env, self.algo.policy, self.algo.env.low_policy, scope=self.algo.scope)
        except AttributeError: # when env doesn't have an inner layer
            parallel_sampler.populate_task(self.algo.env, self.algo.policy)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        # print("obtain samples in batch_polopt")
        cur_params = self.algo.policy.get_param_values()  # a list of numbers
        try:
            cur_low_params = self.algo.low_policy.get_param_values()
            # env_params = cur_low_params if self.algo.train_low else None # need to reset low policy only when training low!
            paths = parallel_sampler.sample_paths(
                policy_params=cur_params,
                low_policy_params=cur_low_params,  # low policy params as env params!
                env_params=[self.algo.env.time_steps_agg, self.algo],  # the parameters to recover for env!
                max_samples=self.algo.batch_size,
                max_path_length=self.algo.max_path_length,
                scope=self.algo.scope,
            )
        except AttributeError:
            paths = parallel_sampler.sample_paths(
                policy_params=cur_params,
                max_samples=self.algo.batch_size,
                max_path_length=self.algo.max_path_length,
                scope=self.algo.scope,
            )
        if self.algo.whole_paths:  # this line is run (whole path)
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            baselinename='linear',
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            discount_low=0.99,
            discount_high=0.9,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            sampler_cls=None,
            sampler_args=None,
            train_low=False,
            train_high=True,
            train_low_with_penalty=False,
            advance_auxilary_reward2=False,
            train_low_with_v_split=False,
            train_low_with_v_gradient=False,
            train_low_with_external=False,
            train_high_every=1,
            time_step_agg_anneal=False,
            anneal_base_number=1.0046158,
            total_low_step=5e4,
            episode_max_low_step=1e3,
            transfer=False,
            transfer_high=False,
            itr_delay=0,
            warm_path=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        # print("init policy") # debug
        self.baseline = baseline
        self.baselinename = baselinename
        self.scope = scope
        self.n_itr = n_itr
        self.current_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.discount_low = discount_low
        self.discount_high = discount_high
        self.train_high_every = train_high_every
        self.gae_lambda = gae_lambda
        # print("gae_lambda", gae_lambda)
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        if sampler_cls is None:
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        ########
        self.train_high = train_high
        self.train_low = train_low
        self.train_low_with_penalty = train_low_with_penalty # train low with penalty when tripped over, no other rewards
        self.train_low_with_v_split = train_low_with_v_split
        self.train_low_with_v_gradient = train_low_with_v_gradient
        self.advance_auxilary_reward2 = advance_auxilary_reward2
        self.train_low_with_external = train_low_with_external
        if self.train_low:
            self.low_policy = self.env.low_policy
            self.low_sampler = LowSampler() # WHAT IS THE USE OF THIS? process low samples
            self.low_sampler.discount = self.discount_low
            self.env.wrapped_env.wrapped_env.set_algo(self)  # add the algorithm to the inner env!
        self.step_anneal = time_step_agg_anneal # specify if the number of low steps in a single high step anneals!
        self.anneal_base_number = anneal_base_number
        self.total_low_step = total_low_step
        self.episode_max_low_step = episode_max_low_step
        self.itr_delay = itr_delay
        self.transfer = transfer
        self.transfer_high = transfer_high
        self.warm_path = warm_path
        global opt_algorithm
        opt_algorithm = self

    def anneal_step_num(self, itr):
        # anneal_base_number : 1.00***
        # 100 * 1.023293 ** -100 = 10
        # 100 * 1.0046158 ** -500 = 10
        self.env.time_steps_agg = int(100*self.anneal_base_number**(-itr)) + 90
        if self.env.time_steps_agg < 10:
            self.env.time_steps_agg = 10
        self.batch_size = self.total_low_step / self.env.time_steps_agg
        self.max_path_length = self.episode_max_low_step / self.env.time_steps_agg

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def test(self, dir): # Rui: just run and render
        self.start_worker()
        logger.set_snapshot_dir(dir)
        self.anneal_step_num(self.current_itr)

        for itr in range(100): # run 100 iterations in total!
            self.n_parallel = 1
            paths = self.sampler.obtain_samples(itr)
            # print("time_step_agg", self.env.time_steps_agg)
            # record value function
            distance_record = []
            pos = []
            for idx, path in enumerate(paths):
                episode_distance = path['env_infos']['last_env_info']['distance'] # a list of distances
                episode_pos = path['env_infos']['last_env_info']['actual_pos']
                # episode_pos = path['env_infos']['full_path']['env_infos']['actual_pos']
                # print("one distance", episode_distance)
                distance_record.append(episode_distance)
                pos.append(episode_pos)
            distance_record = np.concatenate(distance_record)
            pos = np.concatenate(pos)
            # print("distance_record", distance_record)
            high_baselines = [self.baseline.predict(path) for path in paths]
            high_baselines = np.concatenate(high_baselines)
            # print("high_baselines", high_baselines)
            with open(dir + "high_values.pkl", 'wb') as f:
                pickle.dump(high_baselines, f)
            with open(dir + "distance.pkl", "wb") as f:
                pickle.dump(distance_record, f)
            # for v plot: record x and y, and baseline value
            with open(dir + "xyposition.pkl", "wb") as f:
                pickle.dump(pos, f)

            self.env.log_diagnostics(paths)
            self.policy.log_diagnostics(paths)
            self.baseline.log_diagnostics(paths)
            logger.dump_tabular(with_prefix=False)
            self.shutdown_worker()

    def warm_start(self):
        pkl_path = self.warm_path

        data = joblib.load(pkl_path)
        algo1 = data["algo"]

        warm_params = data["policy"].get_params_internal()

        if self.transfer_high == True:
            self.policy.set_params_snn(warm_params)  # looks like these params are saved in the pkl file and will not be overridden

        warm_params_low = algo1.low_policy.get_params_internal()
        self.low_policy.set_params_snn(warm_params_low)
        self.env.low_policy.set_params_snn(
            warm_params_low)  # must use env.low_policy! updating algo.low_policy is useless
        del algo1
        del warm_params_low
        del warm_params
        del data

    def train(self):
        self.start_worker()
        self.init_opt()
        # init_opt for low policy
        if self.train_low:
            self.init_opt_low()
        high_times = 0
        obs_concat = adv_concat = lat_concat = pro_concat = act_concat = np.array([])
        start_i = 0
        #for itr in range(self.current_itr, self.n_itr):
        for itr in range(start_i, self.n_itr):
            gc.collect() # force freeing memory
            if self.transfer and itr == start_i:
                self.warm_start()

            with logger.prefix('itr #%d | ' % itr):
                if self.step_anneal:
                    self.anneal_step_num(itr) # update the step length
                paths = self.sampler.obtain_samples(itr)
                self.discount = self.discount_high # change discount every time we train high-level policy!
                samples_data = self.sampler.process_samples(itr, paths)
                self.log_diagnostics(paths)

                if self.train_high == True: # train the high level policy
                    if self.train_high_every and self.train_high_every != 1:
                        # train high every is the time we train the low level per train of high level
                        if high_times < self.train_high_every:
                            if high_times == 0: # initialize concat vars
                                obs_concat, act_concat, adv_concat = ext.extract(
                                    samples_data,
                                    "observations", "actions", "advantages"
                                )
                                pro_concat = samples_data["agent_infos"]['prob']
                                lat_concat = samples_data["agent_infos"]['latents']
                            else:
                                ## below: how should samples be concatenated:
                                obs_tmp, act_tmp, adv_tmp = ext.extract(
                                    samples_data,
                                    "observations", "actions", "advantages"
                                )
                                pro_tmp = samples_data["agent_infos"]['prob']
                                lat_tmp = samples_data["agent_infos"]['latents']
                                obs_concat = np.concatenate((obs_concat, obs_tmp), axis=0)
                                act_concat = np.concatenate((act_concat, act_tmp), axis=0)
                                adv_concat = np.concatenate((adv_concat, adv_tmp), axis=0)
                                pro_concat = np.concatenate((pro_concat, pro_tmp), axis=0)
                                lat_concat = np.concatenate((lat_concat, lat_tmp), axis=0)

                            ## above: how should samples be concatenated
                        if high_times == self.train_high_every:
                            high_times = 0
                            samples_data_concatenated = {'observations': obs_concat, 'actions': act_concat,
                                                         'advantages': adv_concat, 'agent_infos': {'prob': pro_concat,
                                                                                                   'latents': lat_concat}
                                                         }
                            print("training high policy")
                            self.optimize_policy(itr, samples_data_concatenated)
                        high_times += 1
                    else:
                        self.optimize_policy(itr, samples_data)

                if not self.train_low:
                    pass # not training low policy

                elif self.train_low_with_external:
                    print("training low policy with external rewards only")
                    paths_low = []
                    for idx, path in enumerate(paths):
                        last_low_step_num = len(path["env_infos"]["full_path"]["rewards"][-1])

                        path_low = dict(
                            observations=np.concatenate(path['env_infos']["full_path"]["observations"]),
                            actions=np.concatenate(path['env_infos']["full_path"]["actions"]),
                            rewards=np.concatenate(path['env_infos']["full_path"]["rewards"]),
                        )

                        # WR: trim the observation
                        path_low['observations'] = path_low['observations'][:, :self.low_policy.obs_robot_dim]
                        agent_info_low = dict()
                        for key in path['env_infos']["full_path"]['agent_infos']:
                            agent_info_low[key] = np.concatenate(path['env_infos']["full_path"]['agent_infos'][key])
                        path_low["agent_infos"] = agent_info_low
                        env_info_low = dict()
                        for key in path['env_infos']["full_path"]['env_infos']:
                            # print(key, path)
                            env_info_low[key] = np.concatenate(path['env_infos']["full_path"]["env_infos"][key])
                        path_low["env_infos"] = env_info_low

                        paths_low.append(path_low)
                    real_samples = ext.extract_dict(
                        self.low_sampler.process_samples(itr, paths_low),
                        # I don't need to process the hallucinated samples: the R, A,.. same!
                        "observations", "actions", "advantages", "env_infos", "agent_infos"
                    )
                    real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])
                    self.optimize_policy_low(itr, real_samples)

                elif self.train_low and self.train_low_with_penalty and not self.train_low_with_v_split and not self.train_low_with_v_gradient and not self.advance_auxilary_reward2:
                    # train low with penalty as the only reward; train high with sparse reward
                    print("training low policy with penalty only")
                    # self.discount = self.discount_low
                    paths_low = []
                    for idx, path in enumerate(paths):
                        path_low = dict(
                            observations=np.concatenate(path['env_infos']["full_path"]["observations"]),
                            actions=np.concatenate(path['env_infos']["full_path"]["actions"]),
                            rewards=np.concatenate(path['env_infos']["full_path"]["rewards"]),
                        )

                        rewards_raw = np.concatenate(path['env_infos']["full_path"]["rewards"])
                        if np.sum(path['env_infos']["full_path"]['env_infos'][
                                      'inner_rew']) == 1:  # the episode was successful
                            # the last step should minus the reward of reaching the goal (outer reward)
                            # env_info_low[key] was padded, so the last index is not the actual last step!
                            last_low_step = 0
                            for _ in range(rewards_raw.size - 1, 0, -1):
                                if rewards_raw[_] != 0:
                                    last_low_step = _
                                    break
                            rewards_raw[last_low_step] -= self.env.wrapped_env.wrapped_env.goal_rew
                        path_low['rewards'] = rewards_raw

                        # WR: trim the observation
                        path_low['observations'] = path_low['observations'][:, :self.low_policy.obs_robot_dim]
                        agent_info_low = dict()
                        for key in path['env_infos']["full_path"]['agent_infos']:
                            agent_info_low[key] = np.concatenate(path['env_infos']["full_path"]['agent_infos'][key])
                        path_low["agent_infos"] = agent_info_low

                        env_info_low = dict()
                        for key in path['env_infos']["full_path"]['env_infos']:
                            if key == 'outer_rew':
                                env_info_low[key] = path_low['rewards']
                                # print('env_info reward: ', env_info_low[key])
                            else:
                                env_info_low[key] = np.concatenate(path['env_infos']["full_path"]["env_infos"][key])
                        path_low["env_infos"] = env_info_low

                        paths_low.append(path_low)
                    real_samples = ext.extract_dict(
                        self.low_sampler.process_samples(itr, paths_low),
                        # I don't need to process the hallucinated samples: the R, A,.. same!
                        "observations", "actions", "advantages", "env_infos", "agent_infos"
                    )
                    real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])
                    self.optimize_policy_low(itr, real_samples)

                else:
                    print('ERROR! Unknown training mode. See batch_polopt.py for details.')

                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
                try:
                    params["time_steps_agg"] = self.env.time_steps_agg
                except AttributeError: # don't have this attribute
                    pass
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                # to prevent memory leakage
                # info = psutil.virtual_memory()
                # print ('memory percent', info.percent)
                # if info.percent > 95:
                #     break

                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def init_opt_low(self):
        """
                Initialize the optimization procedure. If using theano / cgt, this may
                include declaring all the variables and compiling functions
                """
        raise NotImplementedError

    def optimize_policy_low(self, its, samples_data):
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
