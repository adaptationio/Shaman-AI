import numpy as np
#from utilities import DataGrabber
#import torch
import random
#from ..common import DataGrabber
class BattleAgents():
    def __init__(self, config):
        self.love = 14
        self.config = config
        self.actions = [1,2,3,4]
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
        self.player = Player(self.config)
        #self.numbergen = NumberGen()

    
    def make_episode(self):
        #NumberGen.to_array(10000000, True)
        #self.numbergen = NumberGen()
        #self.numbergen.to_array(10000000, True)
        #self.state_full = np.load('data/game10000000.npy')
        #self.state_full []
        print('make episode')

    def make_current_state(self, count):
        if self.live:
            self.state = self.live_state
            
        else:
            start = (0+count+self.rand)
            end = (1001+count+self.rand)
            #self.state = self.state_full[start:end]
            
        return self.state

    def get_state(self):
        self.state_current = self.state[-1]
        return self.state_current

    def step(self, action):
        self.count += 1
        self.player.action(self.state_current, action)
        self.make_current_state(self.count)
        state = self.state_maker()
        reward = self.reward(self.get_state(), self.bet_value)
        self.player.balance += reward
        done = self.done(self.count)
        if done:
            self.render()
        
        return state, reward, done


    def reset(self):
        self.player.balance = 0
        self.count = 0
        self.make_episode()
        #if self.eval:
            #self.rand = np.random.random_integers(len(self.state_full / 10 * 9), len(self.state_full))
        #else:
            #self.rand = np.random.random_integers(len(self.state_full / 10 * 9))
        self.state = self.make_current_state(self.count)
        #print(len(self.state))
        state = self.state_maker()
        return state

    def render(self):
        print(self.player.balance)


    def state_maker(self):
        #user = self.player.details(self.count)
        state_details = self.state_details(self.state)
        count = np.array([self.count])
        state = self.data_grabber.flatten(state_details, count)

        return state

    def reward(self, state):
        reward = 0
    

        return reward
    
    def done(self, count):
        if count == 10000:
            
            return True
        else:
            return False 

    def state_details(self, state):
        ind = Indicators()
        details = []
        mean = ind.mean(state)
        median = ind.mean(state)
        black, red = ind.bandr(state)
        details.append([mean, median, black, red])
        return details[0]
         

         
class Player():
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

    def action(self, m_price, action, pm_price):
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

    def action(self, m_price, action, pm_price):
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

         



