
import pyautogui
import random
import time
import datetime

pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()
width = int(screen_width/2) * 0.75
height = int(screen_height/2) * 0.75
mid_width = int(screen_width/2)
mid_height = int(screen_height/2)
print(screen_width, screen_height, mid_width, mid_height, width, height)

# pyautogui.moveTo(mid_width, mid_height)
time.sleep(5)

while True:
    x = random.randint(-width, width)
    y = random.randint(-height, height)
    # pyautogui.moveRel(x, y)
    pyautogui.press('volumedown')
    print("mouse moves to ->", pyautogui.position())
    print("current time   ->", datetime.datetime.now())
    print("----------------------------------------------------------------")
    time.sleep(5)
    # pyautogui.moveTo(mid_width, mid_height)
    pyautogui.press('volumeup')
    time.sleep(5)
