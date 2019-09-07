import numpy as np
import time
from rllab.misc import tensor_utils
from rllab.envs.normalized_env import NormalizedEnv  # this is just to check if the env passed is a normalized maze
from rllab.misc import ext

def rollout(env, agent, max_path_length=np.inf, reset_start_rollout=True, keep_rendered_rgbs=False,
            animated=False, speedup=1):
    """
    :param reset_start_rollout: whether to reset the env when calling this function
    :param keep_rendered_rgbs: whether to keep a list of all rgb_arrays (for future video making)
    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    terminated = []
    if reset_start_rollout:
        o = env.reset()  # otherwise it will never advance!!
    else:
        if isinstance(env, NormalizedEnv):
            o = env.wrapped_env.get_current_obs()
        else:
            o = env.get_current_obs()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    if keep_rendered_rgbs:  # will return a new entry to the path dict with all rendered images
        rendered_rgbs = [env.render(mode='rgb_array')]
    while path_length < max_path_length:
        # print("next_o", len(o))
        # print("env", env)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        # print("next_obs", next_o.shape)
        # print("env", env)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            terminated.append(1)
            break
        terminated.append(0)
        o = next_o
        if keep_rendered_rgbs:  # will return a new entry to the path dict with all rendered images
            rendered_rgbs.append(env.render(mode='rgb_array'))
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    # if animated:   # this is off as in the case of being an inner rollout, it will close the outer renderer!
        # env.render(close=True)

    path_dict = dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),  # here it concatenates all lower-level paths!
        # termination indicates if the rollout was terminated or if we simply reached the limit of steps: important
        # when BOTH happend at the same time, to still be able to know it was the done (for hierarchized envs)
        terminated=tensor_utils.stack_tensor_list(terminated),
    )
    if keep_rendered_rgbs:
        path_dict['rendered_rgbs'] = tensor_utils.stack_tensor_list(rendered_rgbs)

    return path_dict

def process_path(paths, itr, low_sampler):
    paths_low = []
    for idx, path in enumerate(paths):
        obs_shape = path['env_infos']["full_path"]["observations"].shape[2]
        # print("obs_shape", path['env_infos']["full_path"]["observations"].shape)
        act_shape = path['env_infos']["full_path"]["actions"].shape[2]
        path_low = dict(
            observations=path['env_infos']["full_path"]["observations"].reshape([-1, obs_shape]),
            actions=path['env_infos']["full_path"]["actions"].reshape([-1, act_shape]),
            rewards=path['env_infos']["full_path"]["rewards"].reshape([-1]),
        )
        agent_info_low = dict()
        # print("obs_shape", path_low["observations"].shape)
        # print("act_shape", path_low["actions"].shape)
        # print("reward_shape", path_low["rewards"].shape)
        for key in path['env_infos']["full_path"]['agent_infos']:
            # print(key)
            new_shape = path['env_infos']["full_path"]["agent_infos"][key].shape[2]
            agent_info_low[key] = path['env_infos']["full_path"]['agent_infos'][key].reshape([-1, new_shape])
            # print(key, agent_info_low[key].shape)
        path_low["agent_infos"] = agent_info_low
        env_info_low = dict()
        for key in path['env_infos']["full_path"]['env_infos']:
            # print(key, path)
            if key == 'com':
                new_shape = path['env_infos']["full_path"]["env_infos"][key].shape[2]
                env_info_low[key] = path['env_infos']["full_path"]['env_infos'][key].reshape(
                    [-1, new_shape])
            else:
                env_info_low[key] = path['env_infos']["full_path"]['env_infos'][key].reshape(
                    [-1])
        path_low["env_infos"] = env_info_low

        paths_low.append(path_low)
    real_samples = ext.extract_dict(
        low_sampler.process_samples(itr, paths_low),
        # I don't need to process the hallucinated samples: the R, A,.. same!
        "observations", "actions", "advantages", "env_infos", "agent_infos"
    )
    real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])

    return real_samples
