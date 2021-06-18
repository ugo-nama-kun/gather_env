# GatherEnv Swimmer/Ant(+low gear)/Snake
Minimum replication of hierarchical deep reinforcement learning "Gather" environments. Replicated environments do not dependent on rllab and use open-ai gym interface. Then, our environments can run with the latest optimization packages.


Models, environment settings, and codes are inspired by previous creators. Written in papers (such as https://sites.google.com/view/hippo-rl)

![Screen_Shot_2021-03-15_at_14.42.37](/uploads/277229a8947f5ad4b8571c9949a13632/Screen_Shot_2021-03-15_at_14.42.37.png)

## Install
```shell
git clone ssh://git@gitlab:50002/n-yoshida/gather_env.git
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

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.
* Neither the name of the The University of Tokyo or Cognitive Developmental Robotics Lab or Naoto Yoshida, nor the names of its contributors 
  may be used to endorse or promote products derived from this software 
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
