import joblib
from rllab.misc.instrument import stub, run_experiment_lite
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

stub(globals())

# exp setup --------------------------------------------------------
mode = "local"
n_parallel = 1


pkl_path = '/home/lsy/Projects/rllab/data/local/ini-par/30Truetrain_low_v_split_Trueanneal_1parallel_50000LowStepNum_0.95discounthigh_linearbaseline_2019_04_24_23_08_19/params.pkl'

# pkl_path = '/home/wr/rllab/data/local/high-only/Falseanneal_1parallel_1trainhighevery_0.95discounthigh_2019_04_19_21_47_38/params.pkl'

data = joblib.load(pkl_path)

algo = data["algo"]

warm_params = data["policy"].get_params_internal()
algo.policy.set_params_snn(warm_params) # looks like these params are saved in the pkl file and will not be overridden
warm_params_low = algo.low_policy.get_params_internal()

params_env_low_policy = algo.env.low_policy.get_params_internal()
algo.low_policy.set_params_snn(warm_params_low)
algo.env.low_policy.set_params_snn(warm_params_low) # must use env.low_policy! updating algo.low_policy is useless

algo.env.animate = False

algo.batch_size = 5e2
algo.test("/home/lsy/test/")

# algo.sampler.algo.batch_size = 5e3
# algo.test("/home/wr/test/")


pkl_path_xy = '/home/lsy/test/xyposition.pkl'
pkl_path_z = '/home/lsy/test/high_values.pkl'
pos = joblib.load(pkl_path_xy)
x = [xy[0] for xy in pos]
y = [xy[1] for xy in pos]

value = joblib.load(pkl_path_z)#['high_baselines']

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, value, 'gray')
plt.show()