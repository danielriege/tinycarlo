
class RewardHandler():
    '''
    Calculates rewards for environment.
    '''
    def __init__(self, track, car, reward_red, reward_green, use_cte):
        self.reward_red = reward_red
        self.reward_green = reward_green
        self.track = track
        self.car = car
        self.last_cte = 0.0
        self.use_cte = use_cte

    def calc_reward(self, colission_object, cte=None, shaping=None):
        reward = 0
        if colission_object != None:
            if colission_object == 'r':
                reward = self.reward_red
            elif colission_object == 'g':
                reward = self.reward_green
        if cte is None and self.use_cte == True:
            cte = self.track.get_cte(self.car.position)
            self.last_cte = cte
        # if abs(cte) > 240: # fallback if no other done is set
        #     reward = 'done'
        if reward == 0:
            if shaping is not None:
                reward = shaping(cte)

        return reward
    
    def __cte_sparse(self, cte):
        if abs(cte) > 10:
            return 0
        else:
            return 1


