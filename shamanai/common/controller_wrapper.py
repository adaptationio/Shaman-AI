import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from .keyboard import KeyboardController, KeyLogger
from .mouse import MouseController, MouseLogger

class Controller_Gym(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    #Define Actions
    ACTION = [0,1]

    def __init__(self, env):
        self.controlling = True
        self.logging = True
        #self.mouse_logger = MouseLogger()
        #self.mouse_controller = MouseController()
        self.keyboard_logger = KeyLogger()
        #self.keyboard_controller = KeyboardController()
        self.env = env
        self.viewer = None
        self.info = None
        self.reward = None
        self.done = False
        self.state = None
        #self.action_dim = 3
        #self.state_dim = 109
        self.num_envs = 1
        self.num_envs_per_sub_batch = 1
        self.total_pips = []
        #self.player = self.env.player
        #self.pips = self.env.pips
        #self.starter = 0

        # forward or backward in each dimension
        #self.action_space = spaces.Discrete(3)
        self.action_space = self.env.action_space

        # observation is the x, y coordinate of the grid
        #low = np.zeros(0, dtype=int)
        #high =  np.array(1, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        #self.observation_space = spaces.Box(low=-100000, high=100000, shape=(109,))
        self.observation_space = self.env.observation_space
        #print("obs")
        #print (self.observation_space)

        # initial condition
        #self.state = self.env.generate_number()
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        #self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        pass

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #self.state = self.env.generate_number()
        #self.env.display()
        #print(action)
        action = self.keyboard_logger.actions()
        #action = 1
        #self.placement = self.env.placement
        self.next_state, self.reward, self.done, info = self.env.step(action)
        #self.info = 0
        #print(self.reward)
        self.info = { 'pnl':1, 'nav':1, 'costs':1 }
        #self.next_state = self.next_state.tolist()
        #self.total_pips.append(self.pips)
        if self.done:
            pass
        return self.next_state, self.reward, self.done, info

    def reset(self):
        self.state = self.env.reset()
        #self.reward = np.array([reward])
        #self.state = self.state.tolist()
        #self.state = np.array([self.state])
        #self.steps_beyond_done = None
        self.done = False
        #self.done = np.array([self.done])
        return self.state

    def is_game_over(self):
        pass
        return

    def render(self, mode="human", close=False):
        self.env.render()
        #self.env.display()
        pass

        return 

