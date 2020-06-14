"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import retro
from collections import deque
from gym import spaces
import cv2
import tensorflow as tf
import json

from baselines.common.atari_wrappers import WarpFrame, FrameStack
#import gym_remote.client as grc

def make_env(game=None, state=None, stack=False, scale_rew=True, allowbacktrace=False, custom=True):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock')
    #env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    #env = retro.make(game='StreetsOfRage2-Genesis', state='1Player.Axel.Level1')
    #env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')
    #env = retro.make(game='SuperMarioWorld-Snes', state='Bridges1')
    env = retro.make(game=game, state=state)
    #SuperMarioWorld-Snes ['Bridges1',
    env.seed(0)
    env = SonicDiscretizerV3(env)
    #env = StreetOfRage2Discretizer(env)
    #env = StreeFighter2Discretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrameRGB(env)
    if custom:
        env = CustomGym(env)
    if allowbacktrace:
        env = AllowBacktracking(env)
    if stack:
        env = FrameStack(env, 4)
    env = Controller_Gym(env) 
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class SonicDiscretizerV2(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizerV2, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class SonicDiscretizerV3(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizerV3, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [[], ['LEFT'], ['RIGHT'], ['B'], ['RIGHT', 'B'], ['DOWN'], ['DOWN','B'], ['RIGHT','B'], ['DOWN'], ['B'], []]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class StreetOfRage2Discretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(StreetOfRage2Discretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class StreeFighter2Discretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(StreeFighter2Discretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'],
                   ['DOWN', 'B'], ['B'], ['DOWN', 'A'], ['A'], ['DOWN', 'C'], ['C']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class SuperMarioWorldDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SuperMarioWorldDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['RIGHT', 'B'], ['B'], ['RIGHT', 'A'], ['A'],
                   ['RIGHT', 'C'], ['C'], ['RIGHT', 'Y'], ['Y'], ['RIGHT', 'X'], ['X'], ['RIGHT', 'Z'], ['Z']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.005

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self.env.render()
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class CustomGym(gym.Wrapper):
    """
    add custom features
    """
    def __init__(self, env):
        super(CustomGym, self).__init__(env)

    def reset(self, **kwargs): # pylint: disable=E0202
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        self.env.render()
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

class WarpFrameRGB(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
#from keyboard import KeyboardController, KeyLogger
#from mouse import MouseController, MouseLogger

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


from pynput import keyboard
from pynput.keyboard import Key, Controller
moves =[False, False]

class KeyLogger():
    def __init__(self):
        self.moose = "9"
        self.listener = keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.listener.start()
        self.moves = [False, False, False, False, False]
        self.action = 0
    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
            self.moose = key
            print(key)
            if key.char == '1':
                self.moves[1] = True
            if key.char == '0':
                self.moves[0] = True
            if key.char == '2':
                self.moves[2] = True
            if key.char == '3':
                self.moves[3] = True
            if key.char == '4':
                self.moves[4] = True
                
        
        except AttributeError:
            print('special key {0} pressed'.format(
                key))
        return 

    def on_release(self, key):
        print('{0} released'.format(
            key))
        #self.moose = ''
        moose = key
        if key.char == "0":
            self.moves[0] = False
        if key.char == "1":
            self.moves[1] = False
        if key.char == "2":
            self.moves[2] = False
        if key.char == "3":
            self.moves[3] = False
        if key.char == "4":
            self.moves[4] = False
        if key == keyboard.Key.esc:
            # Stop listener
            return False
    def actions(self):
        if self.moves[3] == True and self.moves[2] == True:
            self.action = 4  
        elif self.moves[1] == True:
            self.action = 1
        elif self.moves[2] == True:
            self.action = 2
        elif self.moves[3] == True:
            self.action = 3
        elif self.moves[3] == True:
            self.action = 3
        

        else:
            self.action = 0
        return self.action




class KeyboardController():
    def __init__(self):
        self.moose = "yeah"
        self.keyboard = Controller()
    
    def press(self, key):
        self.keyboard.press(Key.key)

    def release(self, key):
        self.keyboard.release(Key.key)

    def press_release(self, key):
        self.keyboard.press(Key.key)
        self.keyboard.release(Key.key)
    
    def typeing(self, string):
        self.keyboard.type(string)



from pynput import mouse
from pynput.mouse import Button, Controller

class MouseLogger():
    def __init__(self):
        self.test= "test"
        self.listener = mouse.Listener(on_move=self.on_move,on_click=self.on_click,on_scroll=self.on_scroll)
        self.listener()
    def on_move(self, x, y):
        print('Pointer moved to {0}'.format(
            (x, y)))

    def on_click(self, x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False

    def on_scroll(self, x, y, dx, dy):
        print('Scrolled {0} at {1}'.format(
            'down' if dy < 0 else 'up',
            (x, y)))

    def listener_block(self):
        with mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll) as self.listener:
            self.listener.join()

class MouseController():
    def __init__(self):
        pass
        self.mouse = Controller()

    def mouse_position(self):
        # Read pointer position
        print('The current pointer position is {0}'.format(
        self.mouse.position))
        return self.mouse.position
    
    def mouse_postiion_set(self, x, y):

        # Set pointer position
        self.mouse.position = (x, y)
        print('Now we have moved it to {0}'.format(
        self.mouse.position))

    def mouse_move(self, x, y):
        # Move pointer relative to current position
        self.mouse.move(x, y)

    def mouse_press(self, right=False):
        if right == True:
            self.mouse.press(Button.right)
        else:
            self.mouse.press(Button.left)

    def mouse_release(self, right=False):
        if right == True:
            self.mouse.release(Button.right)
        else:
            self.mouse.release(Button.left)

    def mouse_double_click(self, right=False):
        # Double click; this is different from pressing and releasing
        # twice on Mac OSX
        if right == 'right' or right == True:
            self.mouse.click(Button.right, 2)
        else:
            self.mouse.click(Button.right, 2)

    def mouse_scroll(self, x, y):
        # Scroll two steps down
        self.mouse.scroll(x, y)