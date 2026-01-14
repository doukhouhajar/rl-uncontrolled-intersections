'''import os
from typing import Optional

import gym
import numpy as np

from ae.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):
    def __init__(self, env : gym.Env, ae_path: Optional[str] = os.environ["AAE_PATH"]):
        super().__init__(env)
        assert ae_path is not None
        self.ae = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ae.z_size+1,), dtype=np.float32)

    def reset(self):
        # Convert to BGR



        obs=self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1])
        new_obs=np.concatenate(obs.flatten(),[0.0])
    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        
        enc_im= self.ae.encode_from_raw_image(obs[:, :, ::-1])
        speed=infos["speed"]
        new_obs=np.concatenate([enc_im.flatten(),[speed]])
        return enc_im.flatten(), reward, done, infos'''

import os
import cv2
import gym
import numpy as np
from typing import Optional

from ae.autoencoder import load_ae


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
