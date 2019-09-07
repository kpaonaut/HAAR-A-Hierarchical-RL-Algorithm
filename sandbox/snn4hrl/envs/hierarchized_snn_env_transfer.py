class HierarchizedSnnEnvTransfer(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            time_steps_agg=1,
            discrete_actions=True,
            pkl_path=None,
            json_path=None,
            npz_path=None,
            animate=False,
            keep_rendered_rgb=False,
    ):
        """
        :param env: Env to wrap, should have same robot characteristics than env where the policy where pre-trained on
        :param time_steps_agg: Time-steps during which the SNN policy is executed with fixed (discrete) latent
        :param discrete_actions: whether the policy are applied alone or with a linear combination
        :param pkl_path: path to pickled pre-training experiment that includes the pre-trained policy
        :param json_path: path to json of the pre-training experiment. Requires npz_paths of the policy params
        :param npz_path: only required when using json_path
        :param keep_rendered_rgb: the returned frac_paths include all rgb images (for plotting video after)
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env) # the inner env can be referred to as wrapped_env
        self.time_steps_agg = time_steps_agg
        self.discrete_actions = discrete_actions
        self.animate = animate
        self.keep_rendered_rgb = keep_rendered_rgb
        if json_path:
            self.data = json.load(open(os.path.join(config.PROJECT_PATH, json_path), 'r'))
            self.low_policy_latent_dim = self.data['json_args']['policy']['latent_dim']
        elif pkl_path:
            pkl_path = os.path.join(config.PROJECT_PATH, pkl_path)
            self.data = joblib.load(pkl_path)
            self.low_policy_latent_dim = self.data['policy'].latent_dim
        else:
            raise Exception("No path to file given")

        self.low_policy = GaussianMLPPolicy_snn_restorable(
            env_spec=env.spec,
            env=env,
            #latent_dim=latent_dim, # restore from file
            latent_name='categorical',
            pkl_path=pkl_path,
            bilinear_integration=True,  # concatenate also the outer product
            hidden_sizes=(64, 64),
            min_std=1e-6,
        )

    @property
    @overrides
    def action_space(self):
        lat_dim = self.low_policy_latent_dim
        if self.discrete_actions:
            return spaces.Discrete(lat_dim)  # the action is now just a selection
        else:
            ub = 1e6 * np.ones(lat_dim)
            return spaces.Box(-1 * ub, ub)

    #@overrides
    def set_param_values(self, params):
        # Rui: setting env param when n_parallel != 1
        self.time_steps_agg = params[0]
        self.wrapped_env.wrapped_env.algo = params[1]
        #self.low_policy.set_param_values(params) # is set separately!

    @overrides
    def step(self, action):
        #print('!!!!!!!!!!!', self.time_steps_agg)
        action = self.action_space.flatten(action)
        with self.low_policy.fix_latent(action):
            # print("From hier_snn_env --> the hier action is prefixed latent: {}".format(self.low_policy.pre_fix_latent))
            if isinstance(self.wrapped_env, FastMazeEnv):
                with self.wrapped_env.blank_maze():
                    frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                        reset_start_rollout=False, keep_rendered_rgbs=self.keep_rendered_rgb,
                                        animated=self.animate, speedup=1000)
                next_obs = self.wrapped_env.get_current_obs()
                #next_obs = frac_path['observations'][-1]
            elif isinstance(self.wrapped_env, NormalizedEnv) and isinstance(self.wrapped_env.wrapped_env, FastMazeEnv):
                with self.wrapped_env.wrapped_env.blank_maze():
                    # print("max_path_length", self.time_steps_agg)
                    frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                        reset_start_rollout=False, keep_rendered_rgbs=self.keep_rendered_rgb,
                                        animated=self.animate, speedup=1000)
                    # print("low_policy", self.low_policy)
                next_obs = self.wrapped_env.wrapped_env.get_current_obs()
                #next_obs = frac_path['observations'][-1]
                # print("wrapped_env", self.wrapped_env.wrapped_env)
                # print("next_obs_hier", next_obs.shape)
            else:
                frac_path = rollout(self.wrapped_env, self.low_policy, max_path_length=self.time_steps_agg,
                                    reset_start_rollout=False, keep_rendered_rgbs=self.keep_rendered_rgb,
                                    animated=self.animate, speedup=1000)
                next_obs = frac_path['observations'][-1]

            reward = np.sum(frac_path['rewards'])
            terminated = frac_path['terminated'][-1]
            done = self.time_steps_agg > len(
                frac_path['observations']) or terminated  # if the rollout was not maximal it was "done"
            # it would be better to add an extra flagg to this rollout to check if it was done in the last step
            last_agent_info = dict((k, val[-1]) for k, val in frac_path['agent_infos'].items())
            last_env_info = dict((k, val[-1]) for k, val in frac_path['env_infos'].items())
        # print("finished step of {}, with cummulated reward of: {}".format(len(frac_path['observations']), reward))
        if done:
            # if done I need to PAD the tensor so there is no mismatch. Pad with the last elem, but not the env_infos!
            # still padding env_infos, because env_infos not infect training
            frac_path['env_infos'] = tensor_utils.pad_tensor_dict(frac_path['env_infos'], self.time_steps_agg)
            # full_path = tensor_utils.pad_tensor_dict(frac_path, self.time_steps_agg, mode='last')
            # # you might be padding the rewards
            # actual_path_length = len(frac_path['rewards'])
            # full_path['rewards'][actual_path_length:] = 0.
            # print("no padding")
            full_path = frac_path
        else:
            full_path = frac_path
        # print("last_env_info", last_env_info)
        # print("last_agent_info", last_agent_info)
        # print("full_path", full_path)
        return Step(next_obs, reward, done,
                    last_env_info=last_env_info, last_agent_info=last_agent_info, full_path=full_path)
        # the last kwargs will all go to env_info, so path['env_info']['full_path'] gives a dict with the full path!

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        # to use the visualization I need to append all paths
        expanded_paths = [tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path']) for path in paths]
        self.wrapped_env.log_diagnostics(expanded_paths, *args, **kwargs)

    def __str__(self):
        return "Hierarchized: %s" % self._wrapped_env

hierarchize_snn_transfer = HierarchizedSnnEnvTransfer
