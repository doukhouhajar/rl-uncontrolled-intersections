
import gym
import gym_donkeycar
import pickle
from ae.wrapper import AutoencoderWrapper
from sb3_contrib import TQC
MODEL = "../rl-baselines3-zoo/logs/tqc/donkey-mountain-track-v0_20/rl_model_200000_steps.zip"

env = gym.make('donkey-mountain-track-v0')
env=AutoencoderWrapper(env)
model = TQC.load(MODEL, env=env)

buffer = []
episodes = 40

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _ = env.step(action)

        buffer.append((obs, action, reward, next_obs, done))
        obs = next_obs

    print("episode", ep, "done")

with open("buffer.pkl", "wb") as f:
    pickle.dump(buffer, f)

print("buffer saved")
