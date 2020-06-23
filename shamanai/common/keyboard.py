from pynput import keyboard
from pynput.keyboard import Key, Controller
moves =[False, False]

class KeyLogger():
    def __init__(self, listener=True, action_space=5, keys=['0','1','2','3','4'], config=None):
        self.moose = "9"
        self.keys = keys
        self.actions = [False] * action_space
        self.action = 0
        if listener:
            self.listener_start()
    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
            self.moose = key
            print(key)
            if key.char:
                for i in range(len(self.keys)):
                    if key.char == str(self.keys[i]):
                        self.actions[i] = True
            
                
        
        except AttributeError:
            print('special key {0} pressed'.format(
                key))
        return 

    def on_release(self, key):
        print('{0} released'.format(
            key))
        #self.moose = ''
        moose = key
        if key.char:
            for i in range(len(self.keys)):
                    if key.char == str(self.keys[i]):
                        self.actions[i] = False
    
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def action_step(self):
        if self.actions[3] == True and self.actions[2] == True:
            self.action = 4  
        elif self.actions[1] == True:
            self.action = 1
        elif self.actions[2] == True:
            self.action = 2
        elif self.actions[3] == True:
            self.action = 3
        elif self.actions[3] == True:
            self.action = 3
        

        else:
            self.action = 0
        return self.action

    def listener_start(self):
        self.listener = keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.listener.start()

    def listener_stop(self):
        self.listener = keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.listener.stop()





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


#test = KeyLogger()
#moose = True
#while moose:
    #tester = moose
    #print(test.actions)


