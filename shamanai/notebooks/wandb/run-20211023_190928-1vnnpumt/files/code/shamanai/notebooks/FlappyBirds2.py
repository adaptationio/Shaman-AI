#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pyautogui
import imutils
import cv2
import pytesseract
from pynput import mouse
import time
import easyocr
import torch
#from PIL import ImageGrab
import pyscreenshot
import pyscreenshot as ImageGrab
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\adaptation\anaconda3\envs\Shaman-AI\Library\bin\tesseract.exe'
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
from detecto import utils as ut
import optuna

import pandas as pd
import numpy as np

from pathlib import Path
import time

from selenium import webdriver
import cv2
import numpy as np
import numpy as np
import os
import datetime
import csv
import argparse
from functools import partial
from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium.webdriver.common.keys import Keys
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
#options = webdriver.FirefoxOptions()
#options.add_argument('--headless')
#options.add_argument('--no-sandbox')
#options.add_argument('--disable-dev-shm-usage')
from selenium.webdriver.common.by import By

import time


# In[2]:


class ImageProcess():
    def __init__(self):
        self.love = "ramona"
        self.reader = easyocr.Reader(['en'])
        #self.model = core.Model.load('model_weights.pth', ['bird', 'hole'])

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            print('{} at {}'.format('Pressed Left Click' if pressed else 'Released Left Click', (x, y)))
        
            return False # Returning False if you need to stop the program when Left clicked.
        else:
            print('{} at {}'.format('Pressed Right Click' if pressed else 'Released Right Click', (x, y)))
            return False # Returning False if you need to stop the program when Left clicked.

    def get_position(self):
        print("Please select top corner")
        listener = mouse.Listener(on_click=self.on_click)
        listener.start()
        listener.join()
        X=pyautogui.position()
        self.topx = X[0]
        self.topy= X[1]
        time.sleep(1)
        print(self.topx)
        print(self.topy)
        print("Please select bottom corner")
        listener = mouse.Listener(on_click=self.on_click)
        listener.start()
        listener.join()
        B=pyautogui.position()
        self.bottomx = B[0]
        self.bottomy= B[1]
        print(self.bottomx)
        print(self.bottomy)
        return self.topx, self.topy, self.bottomx, self.bottomy

    def object_detection(self):
        state = ImageGrab.grab(bbox=(871, 186, 1227, 821), backend="mss", childprocess=False)
        #state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2GRAY)
        state = cv2.cvtColor(state,cv2.COLOR_GRAY2RGB)
        #state = np.array(state)
        #cv2.imwrite('gray.jpg', state)
        #state = ut.read_image(state) 
        #state= cv2.imread('gray.jpg')
        image = state
        predictions = self.model.predict(image)
        labels, boxes, scores = predictions

        thresh=0.80
        filtered_indices=np.where(scores>thresh)
        filtered_scores=scores[filtered_indices]
        filtered_boxes=boxes[filtered_indices]
        num_list = filtered_indices[0].tolist()
        filtered_labels = [labels[i] for i in num_list]
        #print(filtered_labels)
        if len(filtered_labels) == 2:
            if filtered_labels[0] == "bird":
                bird = filtered_boxes[0]
            else:
                bird = [0,0,0,0]
            if filtered_labels[1] == "hole":
                hole = filtered_boxes[1]
            else:
                hole = [0,0,0,0]
        else:
            try:
                bird = filtered_boxes[0]
                hole = [0,0,0,0]

            except:
                bird = [0,0,0,0]
                hole = [0,0,0,0]
        

        #u = np.concatenate((u), axis=None)
        #m = np.concatenate((m), axis=None)
        #c = np.concatenate((c), axis=None)
        flattened = np.concatenate((bird, hole), axis=None)
        #print(flattened)
        return flattened
        



    def image_ocr(self):
        topx = self.topx
        topy = self.topy
        bottomx = self.bottomx - topx
        bottomy =  self.bottomy - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        #state = cv2.cvtColor(np.array(state), cv2.COLOR_BGR2GRAY)
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(state)
        print(text)
        return text

    def easy_ocr(self):
        topx = self.topx
        topy = self.topy
        bottomx = self.bottomx - topx
        bottomy =  self.bottomy - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        

        #state = 255 - cv2.threshold(state, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Blur and perform text extraction
        #state = cv2.GaussianBlur(state, (3,3), 0)
        #self.reader = easyocr.Reader(['en'])
        result = self.reader.readtext(state)
        return result
    
  
    
    def game_start(self,tx, ty, bx, by):
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        
        

        #state = 255 - cv2.threshold(state, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Blur and perform text extraction
        #state = cv2.GaussianBlur(state, (3,3), 0)
        #self.reader = easyocr.Reader(['en'])
        result = self.reader.readtext(state)
        print(result)
        if len(result) == 1:

            print(result[0][1])
            if result[0][1] == "RESTART":
                return True
            else:
                return False
        else:
            return False

    def game_start_2(self,tx, ty, bx, by, a):
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        #state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = ImageGrab.grab(bbox=(955, 391, 1019, 406), backend="mss", childprocess=False)
        b = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        difference = cv2.subtract(a, b)    
        result = not np.any(difference)
        if result is True:
            return True
        else:
            return False

    def game_start_selenium(self,b, a):
        
        difference = cv2.subtract(a, b)    
        result = not np.any(difference)
        if result is True:
            return True
        else:
            return False
        
    def game_start_make(self,tx, ty, bx, by):
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        np.save('games_start.npy', state)
        return state


    def get_state(self,tx, ty, bx, by):
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        state = ImageGrab.grab(bbox=(753, 186, 1227, 821), backend="mss", childprocess=False)
        #state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        
        state= cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        #cv2.imwrite('red.jpg', state)
        #state = torch.from_numpy(state)
        #state = state.unsqueeze(dim=0)
        return state

    def get_state_save(self,tx, ty, bx, by, count, epoc):
        count = count
        epoc = epoc
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        state = ImageGrab.grab(bbox=(871, 186, 1227, 821), backend="mss", childprocess=False)
        #state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        
        #state= cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        cv2.imwrite('images/flappy/state'+str(count)+str(epoc)+'.jpeg', state)
        #state = torch.from_numpy(state)
        #state = state.unsqueeze(dim=0)
        return state
    
    def get_state_test(self,tx, ty, bx, by):
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        state2= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state2 = cv2.cvtColor(np.array(state2), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        state3= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state3 = cv2.cvtColor(np.array(state3), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        state4= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state4 = cv2.cvtColor(np.array(state4), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        #vis = np.concatenate((state, state2, state3, state4), axis=1)
        vis = np.stack((state, state2, state3, state4))
        #state= cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
        #state = torch.from_numpy(state)
        #state = state.unsqueeze(dim=0)
        return vis

    def get_reward(self,tx, ty, bx, by):
        topx = tx
        topy = ty
        bottomx = bx - topx
        bottomy = by - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        
        

        #state = 255 - cv2.threshold(state, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Blur and perform text extraction
        #state = cv2.GaussianBlur(state, (3,3), 0)
        #self.reader = easyocr.Reader(['en'])
        result = self.reader.readtext(state)
    
        print(result)
        if len(result) == 1:

            return int(0)
        if len(result) == 2:
                if result[1][1] == "I": 
                    return int(1)
                elif result[1][1] == "2":
                    return int(2)
                elif result[1][1] == "8":
                    return int(3)
                else:
                    return(4)
        else:
            return int(0)
        
    def image_ocr_2(self):
        topx = self.topx
        topy = self.topy
        bottomx = self.bottomx - topx
        bottomy =  self.bottomy - topy
        custom_config = r'--oem 3 --psm 10'
        image = pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        gray= cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        thresh = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Blur and perform text extraction
        thresh = cv2.GaussianBlur(thresh, (3,3), 0)
        text = pytesseract.image_to_string(thresh, config=custom_config)
        print(text)
        return text

    def image_stream(self):
        topx = self.topx
        topy = self.topy
        bottomx = self.bottomx - topx
        bottomy =  self.bottomy - topy 
        while True:
            try:
                state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
                state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
                cv2.imshow("Stream", state)
                if cv2.waitKey(1)& 0xFF == ord('q'):
                    break      
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break

    def get_image(self, ):
        topx = self.topx
        topy = self.topy
        bottomx = self.bottomx - topx
        bottomy =  self.bottomy - topy
        state= pyautogui.screenshot(region=(int(topx), int(topy), int(bottomx), int(bottomy)))
        state = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2BGR)
        return state

    def check_image(self, a, b):
        difference = cv2.subtract(a, b)    
        result = not np.any(difference)
        if result is True:
            print("Pictures are the same")
        else:
            print("Pictures are different")




class FlappyBirds2Env():
    def __init__(self):
        self.love = 14
        self.actions = [0,1]
        self.state = None
        self.count = 0
        self.load = True
        self.eval = False
        self.browser = webdriver.Chrome('chromedriver',options=options)
        #self.browser = webdriver.Firefox(executable_path="./drivers/geckodriver",options=options)
        #self.browser = webdriver.Firefox(executable_path="./drivers/geckodriver")
        #self.browser = webdriver.Chrome() 
        self.browser.get("https://flappybird.io/")

# Element to be saved
        self.element = self.browser.find_element(By.ID,'testCanvas')


        time.sleep(5)
        self.element.click()
        time.sleep(2)
        self.image_pro = ImageProcess()
        self.frame = []
        #self.game_start = np.load('games_start.npy')
        #self.game_start2 = pyautogui.pixel(1029, 486)
        #self.game_start = ImageGrab.grab(bbox=(955, 391, 1019, 406), backend="mss", childprocess=False)
        #self.game_start = cv2.cvtColor(np.array(self.game_start), cv2.COLOR_RGB2BGR)
        self.game_start = self.selenium_start()
        self.total_reward = 0
        self.epoc = 0
        #self.emark = (1029, 486)
        #self.pixel = (self.emark[0]-1, self.emark[1]-1, self.emark[0]+1, self.emark[1]+1)
        #self.original_pixel_color = ImageGrab.grab(self.pixel).getpixel((0,0))

    def selenium_grab(self):
        png = self.browser.get_screenshot_as_png()
        location = self.element.location
        size  = self.element.size

        nparr = np.frombuffer(png, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']

        #im = img[left:right, top:bottom]
        im = img[top:int(bottom), left:int(right)]
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        
        im = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        return im

    def selenium_start(self):
        element = self.browser.find_element_by_id("testCanvas")
        png = self.browser.get_screenshot_as_png()
        location = element.location
        size  = element.size

        nparr = np.frombuffer(png, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']

        #im = img[left:right, top:bottom]
        #im = img[top:int(bottom), left:int(right)]
        im = img[460:int(470), 170:int(180)]
        #im = img[460:int(470), 560:int(580)]
        #im = img[440:int(445), 500:int(505)]
        im = np.array(im)
        #im = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        cv2.imwrite('filename.png',im)
        return im

    def selenium_end(self):
        png = self.browser.get_screenshot_as_png()
        location = self.element.location
        size  = self.element.size

        nparr = np.frombuffer(png, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']

        #im = img[left:right, top:bottom]
        #im = img[top:int(bottom), left:int(right)]
        im = img[460:int(416), 170:int(179)]
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        #im = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        cv2.imwrite('filename.png',im)
        return im

    def controller(self):
        jack = "mose"
    
    def state_maker(self):
        #frame = self.image_pro.get_state(752, 188, 1228, 823)
        frame = self.selenium_grab()
        #self.frame.insert(0, frame)
        #elf.frame.pop()
        #state = np.stack((self.frame[0], self.frame[1], self.frame[2], self.frame[3]))
        return frame

    def state_maker_start(self):
        frame = self.image_pro.get_state(752, 188, 1228, 823)
        
        #self.frame.pop()
        return frame

    
        
        rame2 = self.image_pro.get_state(752, 188, 1228, 823)
        frame3 = self.image_pro.get_state(752, 188, 1228, 823)
        frame4 = self.image_pro.get_state(752, 188, 1228, 823)
        self.frame.insert(3, frame)
        self.frame.insert(2, frame2)
        self.frame.insert(1, frame3)
        self.frame.insert(0, frame4)
        state = np.stack((self.frame[0], self.frame[1], self.frame[2], self.frame[3]))

    def timer2(self):
        timeout = time.time() + 0.1   # 5 minutes from now
        while True:
            test = 0
            if test == 5 or time.time() > timeout:
                break
            test = test - 1

    def timer(self):
        timeout = time.time() + 0.325   # 5 minutes from now
        while True:
            test = 0
            if test == 5 or time.time() > timeout:
                break
            test = test - 1
        

    def step(self, action):
        self.count += 1
        
        done = self.check_start()
        if done == True:
            print("done")
        else:
            if action == 1:
                self.element.click()
                #ActionChains(self.browser).send_keys(Keys.SPACE).perform()
                #pyautogui.click(x=795, y= 219, button='left')
                #self.rewards = self.rewards - 0.01
            else:
                d=1
        #self.player.action(self.action)
        #self.timer2()
        #time.sleep(0.1)
        state = self.state_maker()
        #self.image_pro.get_state_save(752, 188, 1228, 823, self.count, self.epoc)
        self.state = state
        #state = self.image_pro.object_detection()
        
        self.rewards = self.count / 100
        self.total_reward = self.total_reward + self.rewards
        
        #reward = self.reward(self.get_state(), self.bet_value)
        #self.player.balance += reward
        remainder = self.count % 1
        #is_divisible = remainder == 0
        #if is_divisible == True:
        if done == False:

            done = self.check_start()

        #if self.count == 100:
            #done = True
        if done == True:
            #self.rewards = self.rewards
            self.total_reward = self.total_reward + self.rewards
            print(self.count)
            print(self.epoc)
            
            
        #time.sleep(0.1)
        
        return state, self.rewards, done

    def check_start(self):
        #check = self.image_pro.game_start_2(955, 391, 1019, 406, self.game_start)
        check = self.image_pro.game_start_selenium(self.selenium_start(), self.game_start)
        #check = pyautogui.pixelMatchesColor(1029, 486,(222, 216, 149))
        #self.new_pixel_color = ImageGrab.grab(pixel).getpixel((0,0))
        #check = self.new_pixel_color == self.original_pixel_color
        return check



    def reset(self):
        
        self.total_reward = 0
        #self.player.balance = 0
        self.count = 0
        self.epoc += 1
        remainder = self.epoc % 100
        is_divisible = remainder == 0
        if is_divisible == True:
            self.browser.quit()
            self.browser = webdriver.Chrome('chromedriver',options=options)
            #self.browser = webdriver.Firefox(executable_path="./drivers/geckodriver",options=options)
            #self.browser = webdriver.Firefox(executable_path="./drivers/geckodriver")
            #self.browser = webdriver.Chrome() 
            self.browser.get("https://flappybird.io/")

            # Element to be saved
            self.element = self.browser.find_element(By.ID,'testCanvas')
            time.sleep(2)
            self.element.click()
            time.sleep(2)
            print("chrome started")
        for i in range(10):
            start = self.check_start()
            if start == True:
                ActionChains(self.browser).send_keys(Keys.SPACE).perform()
                #self.element.click()
                #pyautogui.click(x=914, y=566, button='left')
                #print("start")
                break
            else:
                print("no reset")

        #self.make_episode()
        
        #print(len(self.state))
        #self.timer()
        #time.sleep(1)
        #ActionChains(self.browser).send_keys(Keys.SPACE).perform()
        #self.element.click()
        #self.timer()
        #time.sleep(1)
        #self.element.click()
        #self.timer()
        #time.sleep(1)
        #self.element.click()
        #self.timer()
        #time.sleep(1)
        #self.element.click()
        #self.timer()
        #time.sleep(1)
        #self.element.click()
        #self.timer()
        #time.sleep(1)
        #self.element.click()
        #self.timer()
        #time.sleep(1)
        #self.element.click()
        #time.sleep(.5)
        #self.element.click()
        #time.sleep(.5)
        #self.element.click()
        #time.sleep(.5)
        #self.element.click()
        #time.sleep(.5)
        #self.element.click()
        #time.sleep(.5)
        #self.element.click()
        #time.sleep(.5)
        #self.element.click()
        
        
        
        state = self.state_maker()
        #state = self.image_pro.object_detection()
        self.rewards = 0
        return state

    def render(self):
        print(self.player.balance)

    def reward(self, state):
        reward = 0
    

        return reward
    
    def done(self, count):
        if count == 10000:
            
            return True
        else:
            return False

    def start(self):
        self.something = "lala"


# In[ ]:




# In[4]:


import numpy as np
#from .state import *
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

#MAIN WRAPPER FOR THE ENVIRONMENT TO WORK WITH OPEN AI GYM

class Template_Gym(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    #Define Actions
    ACTION = [0,1]

    def __init__(self):
        #self.config = config
        #self.eval= self.config.eval
        self.start = 0
        self.env = FlappyBirds2Env()
        self.viewer = None
        self.info = None
        self.reward = 0
        self.done = False
        self.state = None
        self.action_dim = 4
        self.state_dim = 32
        self.num_envs = 1
        self.num_envs_per_sub_batch = 1
        self.starter = 0
        self.discrete = True
        
        #self.shape = len(self.env.reset())
        #print(self.shape)
        #self.action_shape = len(self.env.actions)


        #self.df = df
        #self.reward_range = (0, MAX_ACCOUNT_BAL2NCE) 
        if self.discrete:
            # forward or backward in each dimension
            self.action_space = spaces.Discrete(2)
            #self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)

            # observation is the x, y coordinate of the grid
            #low = np.zeros(0, dtype=int)
            #high =  np.array(1, dtype=int) - np.ones(len(self.maze_size), dtype=int)
            #aud = 3339
            #self.observation_space = spaces.Box(low=-10000, high=10000, shape=(3333,))
            self.observation_space = spaces.Box(low=0, high=255,shape=(84,84,1), dtype=np.uint8)
            #self.observation_space = spaces.Box(low=0, high=255, shape=(8,))
        else:
            # Actions of the format Buy x%, Sell x%, Hold, etc.
            self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float32)
            #or
            #self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([3, 1, 1, 1]), dtype=np.float16)


            # Prices contains the OHCL values for the last five prices
            self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6), dtype=np.float16)

        

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
        #self.placement = self.env.placement
        self.next_state, self.reward, self.done= self.env.step(action)
        #self.info = 0
        #print(self.reward)
        self.info = { 'pnl':1, 'nav':1, 'costs':1 }
        #self.next_state = self.next_state.tolist()
        
        #if self.done:
            #print("total pips")
            #print(np.sum(self.total_pips))
            #print(len(self.total_pips))
            #self.starter += 1
            #pass
        return self.next_state, self.reward, self.done, self.info

    def step_async(self, action):
        #self.state = self.env.generate_number()
        #self.env.display()
        #print(action)
        #self.placement = self.env.placement
        self.next_state, self.reward, self.done= self.env.step(action)
        #self.info = 0
        #print(self.reward)
        self.info = { 'pnl':1, 'nav':1, 'costs':1 }
        #self.next_state = self.next_state.tolist()
        
        #if self.done:
            #print("total pips")
            #print(np.sum(self.total_pips))
            #print(len(self.total_pips))
            #self.starter += 1
            #pass
        return self.next_state, self.reward, self.done, self.info

    def step_wait(self):
        #self.state = self.env.generate_number()
        #self.env.display()
        #print(action)
        #self.placement = self.env.placement
        self.next_state, self.reward, self.done= self.env.step(action)
        #self.info = 0
        #print(self.reward)
        self.info = { 'pnl':1, 'nav':1, 'costs':1 }
        #self.next_state = self.next_state.tolist()
        
        #if self.done:
            #print("total pips")
            #print(np.sum(self.total_pips))
            #print(len(self.total_pips))
            #self.starter += 1
            #pass
        return self.next_state, self.reward, self.done, self.info
    

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
        #self.env.display()
        pass

        return 



# In[5]:


import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True
        


# In[ ]:


# In[ ]:




from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def hyper_ppo2():
        return {
            'n_steps': int(128),
            'learning_rate': linear_schedule(2.5e-4),
            'ent_coef': float(0.01),
            'clip_range': linear_schedule(0.1),
            'n_epochs': int(4),
            'batch_size': int(256),
            'vf_coef': float(0.5),
            #'lam': trial.suggest_uniform('lam', 0.8, 1.)
        }

model_params = hyper_ppo2()


# In[7]:



from stable_baselines3.common.utils import set_random_seed
log_dir = "./data/flappyppo/"
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        log_dir = "./data/flappyppo/"
        env = Template_Gym()
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


# In[8]:


import wandb

from wandb.integration.sb3 import WandbCallback


# In[ ]:


#from stable_baselines3.common.cmd_util import make_atari_env
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import VecFrameStack ,DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3 import A2C, PPO, DQN
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
config = {"policy_type": "CnnPolicy", "total_timesteps": 4000000}
experiment_name = f"flappy_{int(time.time())}"
# Initialise a W&B run
wandb.init(
    name=experiment_name,
    project="flappy",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
log_dir = "./data/flappyppo2/"
os.makedirs(log_dir, exist_ok=True)
time.sleep(10)
num_e = 1
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)

env = Template_Gym()
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
#env = Template_Gym()
env_id = "flappy"
num_cpu = 4 # # Number of processes to use
    # Create the vectorized environment
#env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
env = VecFrameStack(env, n_stack=4)
#env = Monitor(env, log_dir)
#env = Controller_Gym(env)
#env = DummyVecEnv([lambda: env])
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./data/ppo2/')
event_callback = EveryNTimesteps(n_steps=5000, callback=checkpoint_on_event)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=1000,
        model_save_path=f"models/{experiment_name}",
    ),
)

#model.learn(total_timesteps=100000)

