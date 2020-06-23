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

