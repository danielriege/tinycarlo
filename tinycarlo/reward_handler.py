

from typing import Collection


class RewardHandler():
    '''
    Calculates rewards for environment.
    '''
    def __init__(self, reward_red, reward_green, reward_tick):
        self.reward_red = reward_red
        self.reward_green = reward_green
        self.reward_tick = reward_tick

    def tick(self, colission_object):
        if colission_object != None:
            if colission_object == 'r':
                return self.reward_red
            elif colission_object == 'g':
                return self.reward_green
        
        return self.reward_tick
