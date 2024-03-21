from gymnasium import Wrapper
import numpy as np
import cv2

class NoiseObservationWrapper(Wrapper):
    """
    Wrapper to add noise to the observation. Only works with observation_space_format="classes".
    """
    def __init__(self, env, blob_max_radius=100, n_blobs=10):
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.max_radius = blob_max_radius
        self.n_blobs = n_blobs
    
    def add_blob_noise_classes(self, observation: np.ndarray):
        for c in range(observation.shape[0]):
            for _ in range(self.n_blobs):
                x, y = np.random.randint(0, observation.shape[2]), np.random.randint(0, observation.shape[1])
                radius = np.random.randint(1, self.max_radius)
                if np.random.choice([True, False], p=[0.3,0.7]):
                    mask = np.zeros(observation[c].shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), radius, 255, -1)
                    mask = cv2.bitwise_and(observation[np.random.randint(0,observation.shape[0])], mask, mask=mask)
                    observation[c] = cv2.bitwise_or(observation[c], mask)
                else:
                    cv2.circle(observation[c], (x, y), radius, 0, -1)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped.observation_space_format == "classes" and not self.env.unwrapped.no_observation:
            observation = self.add_blob_noise_classes(observation)
        return observation, reward, terminated, truncated, info