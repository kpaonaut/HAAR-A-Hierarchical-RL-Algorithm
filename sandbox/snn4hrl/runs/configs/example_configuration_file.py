mode = "local"
train_low = True
train_high = True
train_low_with_penalty = True
n_parallel = 12
maze_id = 8 # small maze
#maze_id = 9 # big maze
fence = False
n_itr = 100 # number of iterations
death_reward = -10.0
# sensor_range = 20.0 # depending on the size of maze, 20 for small, 40 for large
sensor_range = 20.0
low_step_num = 1e3 # 6e5 for large maze, 5e4 for small maze
train_high_every = 1
discount_high = 0.99 # changeable
success_reward = 1000.
exp_prefix_set = 'tmp'
animate = False
time_step_agg_anneal = False
anneal_base_number = 1.023293 # 100 * 1.023293 ** -100 = 10
# anneal_base_number = 1.0046158 # 100 * 1.0046158 ** -500 = 10
max_low_step = 5e4 # num of low steps in one episode, 3000 for large maze
direct_goal = False # whether to include goal as (x,y) or as 20 rays
random_start = True
velocity_field = False # whether to use manually-set velocity field
train_low_with_v_split = True # use HAAR
train_low_with_v_gradient = False
baseline_name = 'linear'
low_level_entropy_penalty = 0.
train_low_with_external = True
itr_delay = 0
transfer = False

pkl_path = path_to_your_pretrained_low_level_skills_stored_in_a_pkl_file
