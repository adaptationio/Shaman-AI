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
        self.action = 0
        self.moves = [False, False, False, False, False]
    def on_move(self, x, y):
        print('Pointer moved to {0}'.format(
            (x, y)))

    def on_click(self, x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        self.moves[1] = True
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