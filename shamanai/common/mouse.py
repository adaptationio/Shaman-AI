from pynput import mouse
from pynput.mouse import Button, Controller
import pyautogui
import imutils
import cv2
import numpy as np

class MouseLogger():
    def __init__(self, listener=True,):
        self.test= "test"
        self.actions = [0,0,0]
        self.xandy = []
        if listener:
            self.listener_start()
    def on_move(self, x, y):
        print('Pointer moved to {0}'.format(
        (x, y)))
        self.actions[0] = x
        self.actions[1] = y
        self.actions[2] = 0

    def on_click(self, x, y, button, pressed):
        print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
        if pressed:
            self.actions[0] = x
            self.actions[1] = y
            self.actions[2] = 1
            
        else:
            self.actions[0] = x
            self.actions[1] = y
            self.actions[2] = 0

    def on_click_record(self, x, y, button, pressed):
        print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
        if pressed:
                self.xandy.append([x,y])
                print("worked")
                print(self.xandy)
        





        

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

    def listener_start(self):
        listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click_record,
            on_scroll=self.on_scroll)
        listener.start()
        #self.listener = mouse.Listener(on_move=self.on_move,on_click=self.on_click,on_scroll=self.on_scroll)
        #self.listener.start()

    def listener_stop(self):
        listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll)
        listener.stop()

    def action_step(self):
        
        print(self.actions)
        return self.actions

    def grab_screen(self):
        while len(self.xandy) < 2:
            pass
        x = self.xandy[0][0]
        y = self.xandy[0][1] 
        w = self.xandy[1][0] - x
        h = self.xandy[1][1] - y
        self.coord = [x,y,w,h]
        image = pyautogui.screenshot(region=(x,y, w, h))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow('HSV image', image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        print("coordniates")
        print(self.coord)


        return self.actions

    def action_step_2(self):
        while len(self.xandy) < 1:
            pass
        

        print(self.xandy[0])
        return self.xandy[0]
        
    

    

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

test = MouseLogger()
moose = True
#while moose:
test.action_step_2()