
from shapely.geometry import LineString, Point

class RewardHandler():
    '''
    Calculates rewards for environment.
    '''
    def __init__(self, track, car, reward_red, reward_green):
        self.reward_red = reward_red
        self.reward_green = reward_green
        self.track = track
        self.car = car
        self.last_cte = 0.0

    def calc_reward(self, colission_object, cte=None, shaping=None):
        reward = 0
        if colission_object != None:
            if colission_object == 'r':
                reward = self.reward_red
            elif colission_object == 'g':
                reward = self.reward_green
        if cte is None:
            cte = self.__cte()
        # if abs(cte) > 240: # fallback if no other done is set
        #     reward = 'done'
        if reward == 0:
            if shaping is not None:
                reward = shaping(cte)

        return reward

    def __cte(self):
        trajectory_points = self.track.gt_clockwise
        if self.car.direction == 1: # counterclockwise
            trajectory_points = self.track.gt_counterclockwise
        trajectory = LineString(trajectory_points)
        position_rear = Point(self.car.position)
        cte = position_rear.distance(trajectory)
        self.last_cte = cte
        return cte
    
    def __cte_sparse(self, cte):
        if abs(cte) > 10:
            return 0
        else:
            return 1


