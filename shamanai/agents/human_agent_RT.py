import numpy as np
from ..common import KeyLogger
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

class HumanAgentRT():
    def __init__(self):
        #self.train = True
        self.ramona = "love"
        self.keyboard_logger = KeyLogger()
        

    def train(self, _obs):
        
        self.action = self.keyboard_logger.action_step()
        print(self.action)
        return self.action


    


    


    
