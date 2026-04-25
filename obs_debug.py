from reproagent.environment import ReproAgentEnv

env = ReproAgentEnv(difficulty='easy', max_steps=10, use_llm=False)
obs, info = env.reset()

print("Checking space bounds:")
for k, space in env.observation_space.spaces.items():
    o = obs[k]
    contains = space.contains(o)
    print(f"{k}: Contains = {contains}")
    if not contains:
        print(f"  Min value: {o.min()}, Max value: {o.max()}")
        print(f"  Space low: {space.low[0]}, Space high: {space.high[0]}")
        print(f"  Is type correct?: {type(o)} == {space.dtype}")
        print(f"  Shape correct?: {o.shape} == {space.shape}")
