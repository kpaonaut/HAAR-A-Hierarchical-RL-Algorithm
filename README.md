# How to run `HAAR`

Welcome to our code release of __Hierachical Reinforcement Learning with Advantage-Based Auxiliary Rewards__, accepted to NeurIPS 2019.

Check out the [videos](http://bit.ly/2JxA0eN) and our [paper]().

We adapted the code from [SNN4HRL](https://github.com/florensacc/snn4hrl) heavily, and also modified some files in rllab. For historical reasons, some folder names remain the same as in SNN4HRL.

You will have to configure [rllab](https://github.com/rll/rllab)(with mujoco) properly. Our configuration is the same as SNN4HRL, so we encourage you to check out their [readme](https://github.com/florensacc/snn4hrl) first and have all the dependencies properly installed.

The first step of running HAAR involves pre-training a set of low-level skills, which can be achieved by using SNN4HRL

~~~
cd rllab/sandbox/snn4hrl
python runs/train_ant_snn.py
~~~

Now you should obtain a `.pkl` file storing the low-level policy.

Change the path `pkl_path` in `rllab/sandbox/snn4hrl/runs/haar_ant_maze.py` to where your pre-trained low-level skills lie (as a pickle file) before running it.

~~~
python runs/haar_ant_maze.py
~~~

This will give you the result of non-annealed HAAR on Ant Maze environment. The result can be found in `rllab/data/local/tmp/exp_name`, as a `.csv` file. The column named `wrapped_succesRate` indicates the success rate of the ant.

If you desire to run annealed HAAR, change the option in the configuration files inside `rllab/sandbox/snn4hrl/runs/configs`. Set `time_step_agg_anneal` to `True`.

You can also transfer both high and low-level skills learned from a previous task to a new task with `transfer.py`.

All experimental results are reproducible with this code release. To reproduce/design different experiments, we encourage you to look at files in `runs/` and tweak the parameters in `configs/`.

If you have any questions, please open an **Issue** on this GitHub page.