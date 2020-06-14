import numpy as np
import pyautogui
import imutils
import cv2
import pytesseract
from bs4 import BeautifulSoup
from selenium import webdriver
import time
%matplotlib auto

games = [1.97]
#for i in range(1000000):
start = 3206947 - 1000
count = 0
driver = webdriver.Firefox()
driver.get('https://www.bustabit.com/game/'+str(start))

time.sleep(5)

for i in range(1000):
    gameid = start + count
    driver.get('https://www.bustabit.com/game/'+str(gameid))
    time.sleep(3)
    image = pyautogui.screenshot(region=(936,475, 290, 30))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
   # image = cv2.GaussianBlur(image, (5, 5), 0)
    #image = cv2.Canny(image, 50, 200, 255)
#image = cv2.medianBlur(image,5)
#cv2.imwrite("i1n_memory_to_disk.png", image)
    #image = cv2.imread("in_memory_to_disk.png")
    #cv2.imshow('HSV image', image); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
#pytesseract.image_to_string(image, lang='eng', \
           #config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    custom_config = r'--oem 1 --psm 1'
    custom_config = r'--oem 3 --psm 6'
    game = pytesseract.image_to_string(image, config=custom_config)
    game = game.replace('x', '')
    game = game.replace('%', '')
    games.append(float(game))
    count = count + 1
    print(game)
    #game = float(game)
    #games = [1.97]
    #if game != games[-1]:
        #games.append(float(game))
        #print(games)

!pip install beautifulsoup4
!pip install selenium

https://github.com/mozilla/geckodriver/releases/download/v0.26.0/geckodriver-v0.26.0-linux64.tar.gz
## Geckodriver
wget https://github.com/mozilla/geckodriver/releases/download/v0.26.0/geckodriver-v0.26.0-linux64.tar.gz
sudo sh -c 'tar -x geckodriver -zf geckodriver-v0.26.0-linux64.tar.gz -O > /usr/bin/geckodriver'
sudo chmod +x /usr/bin/geckodriver
rm geckodriver-v0.23.0-linux64.tar.gz

## Chromedriver
wget https://chromedriver.storage.googleapis.com/2.29/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo chmod +x chromedriver
sudo mv chromedriver /usr/bin/
rm chromedriver_linux64.zip

from bs4 import BeautifulSoup
from selenium import webdriver
import time
driver = webdriver.Firefox()
driver.get('https://www.bustabit.com/game/3206947')

html = driver.page_source
time.sleep(10)
#soup = BeautifulSoup(html)
driver.get('https://www.bustabit.com/game/3206948')
