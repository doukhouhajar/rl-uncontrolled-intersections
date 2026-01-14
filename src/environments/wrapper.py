import os
import cv2
import gym
import numpy as np
from typing import Optional

from src.algorithms.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, ae_path: Optional[str] = os.environ.get("AAE_PATH")):
        super().__init__(env)
        assert ae_path is not None, "AAE_PATH not set"

        self.ae = load_ae(ae_path)

        # latent + speed
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ae.z_size + 1,),
            dtype=np.float32
        )

    def _process(self, img):
        # BGR -> RGB
        img = img[:, :, ::-1]

        # resize to AE input
        img = cv2.resize(img, (96, 96))

        return self.ae.encode_from_raw_image(img).flatten()

    def reset(self):
        obs = self.env.reset()

        z = self._process(obs)
        speed = 0.0

        new_obs = np.concatenate([z, [speed]])
        return new_obs.astype(np.float32)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        z = self._process(obs)
        speed = infos.get("speed", 0.0)

        new_obs = np.concatenate([z, [speed]])

        return new_obs.astype(np.float32), reward, done, infos
