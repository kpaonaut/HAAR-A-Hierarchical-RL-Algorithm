from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np


class LinearFeatureBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val
        # print("coeff", self._coeffs.shape)

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10) # this clipping should be tailored according to obs scale!!
        # print("o", o.shape)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        # print("al", al.shape)
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            # print("coeff", self._coeffs.shape)
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    @overrides
    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        # print("features", self._features(path).shape)
        # print("coeff", self._coeffs.shape)
        return self._features(path).dot(self._coeffs)
