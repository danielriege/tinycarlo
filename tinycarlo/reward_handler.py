
class RewardHandler():
    '''
    Calculates rewards for environment.
    '''
    def __init__(self, track, car, reward_obstacles, use_cte):
        self.track = track
        self.car = car
        self.last_cte = 0.0
        self.use_cte = use_cte
        self.reward_obstacles = reward_obstacles

    def calc_reward(self, colission_object, cte=None, shaping=None):
        reward = 0
        if colission_object != None:
            for obstacle in self.reward_obstacles:
                if colission_object == obstacle['color']:
                    reward = obstacle['reward']
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


