import gym
import gym_donkeycar
from d3rlpy.algos import CQL
from ae.wrapper import AutoencoderWrapper
MODEL = "../d3rlpy_logs/cql_donkey_20260114115150/model_650.pt"
ENV = "donkey-mountain-track-v0"

env = gym.make(ENV)
env=AutoencoderWrapper(env)
# Create model with SAME hyperparams

model = CQL(
    batch_size=512,
    conservative_weight=10.0,
    n_action_samples=20,
    use_gpu=False   # FORCE CPU
)


# 🔑 BUILD NETWORK FIRST
model.build_with_env(env)

# 🔑 NOW load weights
model.load_model(MODEL)

total_reward=0
obs = env.reset()

i=0
while i<5:
    action = model.predict(obs[None])[0]
    obs, reward, done, _ = env.step(action)
    env.render()
    total_reward+=reward
    if done  :
        obs =env.reset()
        i+=1
env.close()

print("Episode reward:", total_reward)
