
import gym
import gym_donkeycar
from sb3_contrib import TQC
import time
from aae_train_donkeycar.ae.wrapper import AutoencoderWrapper
ENV_NAME = "donkey-mountain-track-v0"
MODEL_PATH = "rl-baselines3-zoo/logs/tqc/donkey-mountain-track-v0_20/rl_model_200000_steps"

env = gym.make(ENV_NAME)
env=AutoencoderWrapper(env)
print("Loading model...")
model = TQC.load(MODEL_PATH, env=env, device="cuda")

obs = env.reset()
lap = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Show laps only in terminal
    if "lap" in info:
        lap += 1
        print(f"🏁 LAP {lap}")

    if done:
        print("Episode finished, resetting...")
        obs = env.reset()

    time.sleep(0.05)  # slow for visibility
