# GatherEnv Swimmer/Ant(+low gear)/Snake
Minimum replication of hierarchical deep reinforcement learning "Gather" environments. Replicated environments do not depend on rllab and use open-ai gym interface. Then, our environments can run with the latest optimization packages.


Models, environment settings, and codes are inspired by previous creators. Written in papers (such as https://sites.google.com/view/hippo-rl)

![ant](https://user-images.githubusercontent.com/1684732/122540133-47ceb180-d063-11eb-8239-6aa1ec40d984.png) ![snake](https://user-images.githubusercontent.com/1684732/122540145-4bfacf00-d063-11eb-96c6-5d4b5c29018d.png)


## Install
```shell
git clone git@github.com:ugo-nama-kun/gather_env.git 
pip install -e gather_env
```

## Usage
```python
import gym

env = gym.make("gather_env:LowGearAntGather-v0")

done = False

while not done:
    
    action = env.action_space.sample()
    
    obs, reward, done, info = env.step(action)
```

## Environment List
```shell
# default environments
LowGearAntGather-v0
SnakeGather-v0

# optional
AntGather-v0
SwimmerGather-v0
```

### Requirements
[gym](https://github.com/openai/gym) >= 0.18.0\
[mujoco-py](https://github.com/openai/mujoco-py) >= 2.0.2.13\
[mujoco](https://www.roboti.us/index.html) >= 1.5 located at ~/.mujoco

Tested with python 3.7


### tips
- There are two variation of ant gather environment (low-gear/normal)
- Snake gather environment uses ego-centric-position observation instead of the global position. This might be actually used in the previous study.
- sensor_span param is default 2*math.pi

## License

Copyright (c) 2021-Present, Naoto Yoshida @ Cognitive Developmental Robotics Lab, The University of Tokyo <br>
All rights reserved.
