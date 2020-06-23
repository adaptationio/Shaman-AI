import numpy as np
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

class HumanAgent():
    def __init__(self):
        #self.train = True
        self.ramona = "love"
        
        

    def train(self, _obs):
        
        self.action = input('action')
        print(self.action)
        return self.action


    