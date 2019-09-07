from rllab.misc import ext
from rllab.misc import ext1
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv
from sandbox.snn4hrl.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from rllab.envs.normalized_env import normalize


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_low=None,
            optimizer_args=None,
            step_size=0.01,
            low_level_entropy_penalty=0.0,
            truncate_local_is_ratio=None,
            **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
            # optimizer_low = PenaltyLbfgsOptimizer(**optimizer_args)
        print("optimizer", optimizer)
        # print("optimizer_low", optimizer_low)
        self.optimizer = optimizer
        self.optimizer_low = optimizer_low
        self.step_size = step_size
        self.low_level_entropy_penalty = low_level_entropy_penalty
        self.truncate_local_is_ratio = truncate_local_is_ratio
        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        # print("old_dist_origin", old_dist_info_vars)
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)
        # print("original_input", input_list)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        # print("all_input_values", all_input_values)
        # loss_before = self.optimizer.loss(all_input_values)
        # mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        # mean_kl = self.optimizer.constraint_val(all_input_values)
        # loss_after = self.optimizer.loss(all_input_values)
        # logger.record_tabular('LossBefore', loss_before)
        # logger.record_tabular('LossAfter', loss_after)
        # logger.record_tabular('MeanKLBefore', mean_kl_before)
        # logger.record_tabular('MeanKL', mean_kl)
        # logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        if self.train_low:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
                env=self.env,
                low_policy=self.low_policy,  # Rui: also save low-policy
            )
        else:
            return dict(
                itr=itr,
                policy=self.policy,
                baseline=self.baseline,
                env=self.env,
                # low_policy=self.low_policy,  # Rui: also save low-policy
            )

    #  for train low policy? add init_opt_low
    def init_opt_low(self):
        assert not self.low_policy.recurrent
        is_recurrent = int(self.low_policy.recurrent)
        env = normalize(AntEnv(ego_obs=True))  # WR: AntEnv or AntMazeEnv?

        obs_var_backup = env.observation_space.new_tensor_variable(
            'obs_low',
            extra_dims=1 + is_recurrent,
        )  # 47 dim

        obs_var = env.wrapped_env.observation_space.new_tensor_variable(
            'obs_low',
            extra_dims=1 + is_recurrent,
        )  # 27 dim

        action_var = env.action_space.new_tensor_variable(
            'action_low',
            extra_dims=1 + is_recurrent,
        )
        # print("low_policy", self.low_policy)
        latent_var = self.low_policy.latent_space.new_tensor_variable(
            'latents_low',
            extra_dims=1 + is_recurrent,
        )

        advantage_var = ext1.new_tensor(
            'advantage_low',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.low_policy.distribution  # this can still be the dist P(a|s,__h__)
        old_dist_info_vars = {
            k: ext1.new_tensor(
                'old_%s' % k,  # define tensors old_mean and old_log_std
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        # print("old_dist", old_dist_info_vars)
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]  ##put 2 tensors above in a list

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.low_policy.dist_info_sym(obs_var, latent_var)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            # surr_loss = -TT.mean(lr)
            # surr_loss = TT.mean(advantage_var)
            surr_loss = - TT.mean(lr * advantage_var)

        loss = surr_loss
        # add the entropy penalty,if the penalty is 0 then no penalty on entropy

        loss += self.low_level_entropy_penalty * TT.mean(dist.entropy_sym(dist_info_vars))

        input_list = [  # these are sym var. the inputs in optimize_policy have to be in same order!
                         obs_var,
                         action_var,
                         advantage_var,
                         latent_var,
                     ] + old_dist_info_vars_list  # provide old mean and var, for the new states as they were sampled from it!
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer_low.update_opt(
            loss=loss,
            target=self.low_policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def optimize_policy_low(self, itr, samples_data):
        # raise NotImplementedError
        all_input_values = tuple(ext.extract(  # it will be in agent_infos!!! under key "latents"
            samples_data,
            "observations", "actions", "advantages"
        ))
        # for _ in range(10):
        #     print("OK!")
        # print(samples_data["advantages"])
        agent_infos = samples_data["agent_infos"]
        # for _ in range(10):
        #     print("OK!")
        # print(agent_infos)
        all_input_values += (agent_infos[
                                 "latents"],)  # latents has already been processed and is the concat of all latents, but keeps key "latents"
        info_list = [agent_infos[k] for k in
                     self.low_policy.distribution.dist_info_keys]  # these are the mean and var used at rollout, corresponding to
        all_input_values += tuple(info_list)  # old_dist_info_vars_list as symbolic var
        if self.low_policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer_low.loss(all_input_values)

        # this should always be 0. If it's not there is a problem.
        mean_kl_before = self.optimizer_low.constraint_val(all_input_values)
        logger.record_tabular('MeanKL_Before_low', mean_kl_before)

        with logger.prefix(' Low_PolicyOptimize | '):
            self.optimizer_low.optimize(all_input_values)

        mean_kl = self.optimizer_low.constraint_val(all_input_values)
        loss_after = self.optimizer_low.loss(all_input_values)
        logger.record_tabular('LossBefore_low', loss_before)
        logger.record_tabular('LossAfter_low', loss_after)
        logger.record_tabular('LossBefore_low', loss_before)
        logger.record_tabular('MeanKL_low', mean_kl)
        logger.record_tabular('dLoss_low', loss_before - loss_after)
        return dict()
