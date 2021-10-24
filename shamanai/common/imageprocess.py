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


import numpy as np
import os
import datetime
import csv
import argparse
from functools import partial




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