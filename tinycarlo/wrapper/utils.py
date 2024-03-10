from typing import Dict

def sparse_reward(conditions: Dict[str, bool], sparse_rewards: Dict[str, float]) -> float:
    """
    Calculates the sparse reward based on the given conditions and sparse rewards.

    Args:
        conditions (Dict[str, bool]): A dictionary of condition names and their corresponding boolean values.
        sparse_rewards (Dict[str, float]): A dictionary of condition names and their corresponding sparse reward values.

    Returns:
        float: The calculated sparse reward.

    """
    reward: float = 0.0
    for condition_name, condition in conditions.items():
        if condition and condition_name in sparse_rewards:
            reward += sparse_rewards[condition_name]
    return reward

def linear_reward(x: float, max_x: float, max_reward: float = 1.0) -> float:
    """
    Calculates the linear reward based on the input value.

    Args:
        x (float): The input value.
        max_x (float): The maximum input value which still generates a reward > 0.
        max_reward (float): The maximum reward value. Defaults to 1.0.

    Returns:
        float: The calculated linear reward.
    """
    y = (-max_reward/max_x) * abs(x) + max_reward
    if max_reward > 0:
        return max(y, 0)
    else:
        return min(y, 0)