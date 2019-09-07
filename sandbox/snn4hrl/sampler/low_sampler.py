

import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import joblib

from rllab import config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.config_personal import *
from rllab.envs.normalized_env import normalize
from sandbox.snn4hrl.envs.hierarchized_snn_env import hierarchize_snn
from sandbox.snn4hrl.envs.mujoco.ant_env import AntEnv


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class LowSampler(Sampler):
    def __init__(self):
        """
        :type algo: BatchPolopt
        """
        env_low = normalize(AntEnv(ego_obs=True))
        # baseline_low = LinearFeatureBaseline(env_spec=env_low.spec)
        # low_policy = env.low_policy
        pkl_path = '/home/lsy/Desktop/rllab/data/local/Ant-snn1000/Ant-snn_10MI_5grid_6latCat_bil_0040/params.pkl'
        data = joblib.load(os.path.join(config.PROJECT_PATH, pkl_path))
        low_policy = data['policy']


        self.baseline = LinearFeatureBaseline(env_spec=env_low.spec)
        self.discount = 0.99
        self.gae_lambda = 1.0
        self.center_adv = True
        self.positive_adv = False
        self.policy = low_policy

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        if hasattr(self.baseline, "predict_n"):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        # if not self.algo.policy.recurrent:
        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        if self.center_adv:
            advantages = util.center_advantages(advantages)

        if self.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.mean(self.policy.distribution.entropy(agent_infos))

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )


        logger.log("fitting baseline...")
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)
        logger.log("fitted")

        with logger.tabular_prefix('Low_'):
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('AverageDiscountedReturn',
                                  average_discounted_return)
            logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular('ExplainedVariance', ev)
            logger.record_tabular('NumTrajs', len(paths))
            logger.record_tabular('Entropy', ent)
            logger.record_tabular('Perplexity', np.exp(ent))
            logger.record_tabular('StdReturn', np.std(undiscounted_returns))
            logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
