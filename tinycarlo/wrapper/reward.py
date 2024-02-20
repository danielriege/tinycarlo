from gymnasium import Wrapper
from tinycarlo.wrapper.utils import sparse_reward, linear_reward
from typing import Dict

class LanelineSparseRewardWrapper(Wrapper):
    def __init__(self, env, sparse_rewards: Dict[str, float]):
        """
        A wrapper class that adds sparse rewards based on laneline crossing/touching.
        Depending on the values given in sparse_rewards, the reward is given when the coresponding laneline is crossed/touched.

        Args:
            env (gym.Env): The environment to wrap.
            sparse_rewards (Dict[str, float]): A dictionary mapping laneline types to their corresponding sparse rewards.
        """
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.sparse_rewards = sparse_rewards

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        conditions = {layer_name: info["laneline_distances"][layer_name] < self.env.car.track_width * 2/3 for layer_name in info["laneline_distances"]}
        reward += sparse_reward(conditions, self.sparse_rewards)
        return observation, reward, terminated, truncated, info
    
class LanelineLinearRewardWrapper(Wrapper):
    def __init__(self, env, max_rewards: Dict[str, float]):
        """
        A wrapper class that adds linear rewards based on laneline distances from the car.

        Args:
            env (gym.Env): The environment to wrap.
            max_rewards (Dict[str, float]): A dictionary mapping laneline types to their maximum rewards.
        """
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.max_rewards = max_rewards

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        for layer_name, distance in info["laneline_distances"].items():
            reward += linear_reward(distance, self.env.car.track_width, self.max_rewards[layer_name])
        return observation, reward, terminated, truncated, info
    
class CTESparseRewardWrapper(Wrapper):
    def __init__(self, env, min_cte: float, sparse_reward: float = 1.0):
        """
        A wrapper class that adds a sparse reward based on the cross-track error (CTE) of the car to the lane path.

        Args:
            env (gym.Env): The environment to wrap.
            min_cte (float): The minimum acceptable CTE value in meters
            sparse_reward (float, optional): The reward value to add when the CTE is below the minimum. Defaults to 1.0.
        """
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.min_cte = min_cte
        self.sparse_reward = sparse_reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward += sparse_reward({"cte": abs(info["cte"]) <= self.min_cte}, {"cte": self.sparse_reward})
        return observation, reward, terminated, truncated, info
    
class CTELinearRewardWrapper(Wrapper):
    def __init__(self, env, min_cte: float, max_reward: float = 1.0):
        """
        Wrapper class that adds a linear reward based on the cross-track error (CTE) of the car to the lane path.

        Args:
            env (gym.Env): The environment to wrap.
            min_cte (float): The minimum cross-track error value in meters which is > 0.
            max_reward (float, optional): The maximum reward value. Defaults to 1.0.
        """
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.min_cte = min_cte
        self.max_reward = max_reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward += linear_reward(info["cte"], self.min_cte, self.max_reward)
        return observation, reward, terminated, truncated, info