# HAG-RL: Human Attention Guided Reinforcement Learning for Autonomous Driving in CARLA


## Getting started
1. Install the CARLA simulator, with referring to
https://carla.readthedocs.io/en/latest/start_quickstart/#a-debian-carla-installation

2. Install the dependent package
```shell
pip install -r requirements.txt
```
3. Training the RL agent in the left-turn scenario
```
python train_leftturn.py
--algorithm 0 
--human_model 
--reward_shaping 0 
--seed 123 
--maximum_episode 400 
--initial_exploration_rate 0.5 
--cutoff_exploration_rate 0.05 
--exploration_decay_rate 0.99988 
--warmup 
--warmup_threshold 1e4 
--device cuda 
--simulator_port 2000 
--simulator_render_frequency 12 
--joystick_enabled
```


## Training performance

<img src="figures/results.png" width = "500" height = "400" alt=" " align=center />

(a-b) Results in the left-turn scenario;
(c-d) Results in the congestion scenario.

This repo have some implementation adopted from Prioritized Experience-Based Reinforcement Learning With Human Guidance for Autonomous Driving.

## License
This repo is released under GNU GPLv3 License.


