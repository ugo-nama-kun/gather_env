from gather_env.low_gear_ant_gather_env import LowGearAntGatherEnv

print("start")
env_batch = []
for indx in range(10):
    env_batch.append((indx, LowGearAntGatherEnv()))

for ev in env_batch:
    ev[1].reset()
for t in range(10):
    for i, ev in enumerate(env_batch):
        print(t, i)
        ev[1].step(ev[1].action_space.sample())

print("done")