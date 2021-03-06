import numpy as np
#from utilities import DataGrabber
#import torch
import random
#from ..common import DataGrabber
class BattleAgents():
    def __init__(self, config):
        self.love = 14
        self.config = config
        self.actions = [0,1,2]
        self.state = None
        self.state_full = None
        self.state_current = None
        self.count = 0
        self.diff = 0
        self.load = True
        self.eval = False
        self.bet_value = 2
        self.live_state =[2]
        self.live = True
        self.player1 = Player(self.config)
        self.player2 = Player(self.config)
        #self.numbergen = NumberGen()

    
    def make_state(self):
        state = [0,1,2,3,4,5,6,7,8,9]
        return state

    def make_current_state(self, count):
        count = count
        self.state = count
            
        return self.state

    def get_state(self):
        self.state_current = self.state[-1]
        return self.state_current

    def step(self, action):
        self.count += 1
        action = action
        if self.state[0] == action:
            self.player1.score += 1
        #self.player.action = int(input("action 0,1,2"))
        reward = self.reward(self.state, action)
        #self.make_current_state(self.count)
        self.state = self.state_maker()
        print(self.state)
        #self.player.balance += reward
        done = self.done(self.count)
        if done:
            self.render()
        
        return self.state, reward, done


    def reset(self):
        self.player1.score = 0
        #self.player2.hp = 100
        self.count = 0
        #self.make_episode()
        #if self.eval:
            #self.rand = np.random.random_integers(len(self.state_full / 10 * 9), len(self.state_full))
        #else:
            #self.rand = np.random.random_integers(len(self.state_full / 10 * 9))
        #self.state = self.make_current_state(self.count)
        #print(len(self.state))
        self.state = self.state_maker()
        print(self.state)
        return self.state

    def render(self):
        print(self.player1.score)


    def state_maker(self):
        #user = self.player.details(self.count)
        #state_details = self.state_details(self.state)
        #count = np.array([self.count])
        #state = self.data_grabber.flatten(state_details, count)
        self.count
        state = [0,1,2]
        state = [state[np.random.randint(0,3)]]

        state = np.asarray(state)
        return state

    def reward(self, state, action):
        if state[0] == action:
            reward = 1
        else:
            reward = 0
    

        return reward
    
    def done(self, count):
        if count == 10 or self.player1.hp < 0:
            print(self.player1.score)
            
            return True
        else:
            return False 

    def state_details(self, state):
        details = []
        details.append([self.state])
        return details[0]
         

         
class Player():
    def __init__(self, config):
        self.config = config
        self.hp = 1000
        self.action = 0
        self.details = "details"
        self.score = 0
    
    def action_user(self):
        #print(len)
        #self.update(m_price)
        x = input('buy, sell, close, hold?:')
        x = str(x)
        if x == "buy":
            self.open_position_long(m_price, pm_price)
        elif  x == "sell":
            self.open_position_short(m_price, pm_price)
        elif x == "close":
            self.close_position(m_price, pm_price)
        elif x == "hold":
            self.hold_position(m_price, pm_price)
        else:
            self.hold_position(m_price, pm_price)

    def actions(self, m_price, action, pm_price):
        #print(len)
        #self.update(m_price)
        x = action
        #x = self.normalize(x)
        #x = int(x)
        #if self.placement < -0.200 or self.placement > 0.200:
            #self.close_position(m_price, pm_price)
        if self.config.buy == True:
        
            if x == 2:
                self.open_position_long(m_price, pm_price, x)
            elif x == 1:
                self.close_position(m_price, pm_price, x)
            elif x == 0:
                self.open_position_short(m_price, pm_price, x)
        else:   
            if x == 0:
                self.open_position_short(m_price, pm_price, x)
            elif x == 1:
                self.close_position(m_price, pm_price, x)
            elif x == 2:
                self.open_position_long(m_price, pm_price, x)
        if x == 3:
            self.hold_position(m_price, pm_price)

class PlayerAI():
    def __init__(self, config):
        self.config = config
        

    def action_user(self, m_price, pm_price):
        #print(len)
        #self.update(m_price)
        x = input('buy, sell, close, hold?:')
        x = str(x)
        if x == "buy":
            self.open_position_long(m_price, pm_price)
        elif  x == "sell":
            self.open_position_short(m_price, pm_price)
        elif x == "close":
            self.close_position(m_price, pm_price)
        elif x == "hold":
            self.hold_position(m_price, pm_price)
        else:
            self.hold_position(m_price, pm_price)

    def actions(self, m_price, action, pm_price):
        #print(len)
        #self.update(m_price)
        x = action
        #x = self.normalize(x)
        #x = int(x)
        #if self.placement < -0.200 or self.placement > 0.200:
            #self.close_position(m_price, pm_price)
        if self.config.buy == True:
        
            if x == 2:
                self.open_position_long(m_price, pm_price, x)
            elif x == 1:
                self.close_position(m_price, pm_price, x)
            elif x == 0:
                self.open_position_short(m_price, pm_price, x)
        else:   
            if x == 0:
                self.open_position_short(m_price, pm_price, x)
            elif x == 1:
                self.close_position(m_price, pm_price, x)
            elif x == 2:
                self.open_position_long(m_price, pm_price, x)
        if x == 3:
            self.hold_position(m_price, pm_price)

         



