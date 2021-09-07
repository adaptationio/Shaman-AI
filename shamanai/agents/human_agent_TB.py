import numpy as np
#from stable_baselines import DQN
#from stable_baselines.gail import generate_expert_traj

class HumanAgentTB():
    def __init__(self):
        #self.train = True
        self.ramona = "love"
        
        

    def train(self, _obs):
        
        self.action = input('action')
        self.action = int(self.action)
        print(self.action)
        return self.action


    