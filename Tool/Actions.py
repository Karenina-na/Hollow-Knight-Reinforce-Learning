# Define the actions we may need during training
# You can define your actions here
import ctypes

from Tool.SendKey import PressKey, ReleaseKey
from Tool.WindowsAPI import grab_screen
import time
import cv2
import threading
import win32gui, win32ui, win32con

# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
UP_ARROW = 0x57  # w
DOWN_ARROW = 0x53  # s
LEFT_ARROW = 0x41  # A
RIGHT_ARROW = 0x44  # D

L_SHIFT = 0xA0
N = 0x4E
L = 0x4C
J = 0x4A
K = 0x4B


# move actions
# 0
def Nothing():
    ReleaseKey(LEFT_ARROW)
    ReleaseKey(RIGHT_ARROW)
    pass


# Move
# 0
def Move_Left():
    PressKey(LEFT_ARROW)
    time.sleep(0.01)


# 1
def Move_Right():
    PressKey(RIGHT_ARROW)
    time.sleep(0.01)


# 2
def Turn_Left():
    PressKey(LEFT_ARROW)
    time.sleep(0.01)
    ReleaseKey(LEFT_ARROW)


# 3
def Turn_Right():
    PressKey(RIGHT_ARROW)
    time.sleep(0.01)
    ReleaseKey(RIGHT_ARROW)


# ----------------------------------------------------------------------

# other actions
# Attack
# 0
def Attack():
    PressKey(J)
    time.sleep(0.15)
    ReleaseKey(J)
    Nothing()
    time.sleep(0.01)


# 1
# def Attack_Down():
#     PressKey(DOWN_ARROW)
#     PressKey(X)
#     time.sleep(0.05)l
#     ReleaseKey(X)
#     ReleaseKey(DOWN_ARROW)
#     time.sleep(0.01)
# 1
def Attack_Up():
    # print("Attack up--->")
    PressKey(UP_ARROW)
    PressKey(J)
    time.sleep(0.11)
    ReleaseKey(J)
    ReleaseKey(UP_ARROW)
    Nothing()
    time.sleep(0.01)


# JUMP
# 2
def Short_Jump():
    PressKey(L)
    PressKey(DOWN_ARROW)
    PressKey(J)
    time.sleep(0.2)
    ReleaseKey(J)
    ReleaseKey(DOWN_ARROW)
    ReleaseKey(L)
    Nothing()


# 3
def Mid_Jump():
    PressKey(L)
    time.sleep(0.2)
    PressKey(J)
    time.sleep(0.2)
    ReleaseKey(J)
    ReleaseKey(L)
    Nothing()


# Skill
# 4
# def Skill():
#     PressKey(Z)
#     PressKey(X)
#     time.sleep(0.1)
#     ReleaseKey(Z)
#     ReleaseKey(X)
#     time.sleep(0.01)
# 4
def Skill_Up():
    PressKey(UP_ARROW)
    PressKey(K)
    PressKey(J)
    time.sleep(0.15)
    ReleaseKey(UP_ARROW)
    ReleaseKey(K)
    ReleaseKey(J)
    Nothing()
    time.sleep(0.15)


# 5
def Skill_Down():
    PressKey(DOWN_ARROW)
    PressKey(K)
    PressKey(J)
    time.sleep(0.2)
    ReleaseKey(J)
    ReleaseKey(DOWN_ARROW)
    ReleaseKey(K)
    Nothing()
    time.sleep(0.3)


# Rush
# 6
def Rush():
    PressKey(L_SHIFT)
    time.sleep(0.1)
    ReleaseKey(L_SHIFT)
    Nothing()
    PressKey(J)
    time.sleep(0.03)
    ReleaseKey(J)


# Cure
# 7
def Cure():
    PressKey(N)
    time.sleep(1.4)
    ReleaseKey(N)
    time.sleep(0.1)


# Restart function
# it restart a new game
# it is not in actions space
def Look_up():
    PressKey(UP_ARROW)
    time.sleep(0.1)
    ReleaseKey(UP_ARROW)





def restart(state_size):
    while True:
        station = cv2.resize(cv2.cvtColor(grab_screen(state_size), cv2.COLOR_RGBA2RGB), (1000, 500))
        if station[187][300][0] != 0:
            time.sleep(1)
        else:
            break
    time.sleep(1)
    Look_up()
    time.sleep(3)
    Look_up()
    time.sleep(1)
    # PressKey(DOWN_ARROW)
    # time.sleep(0.1)
    # ReleaseKey(DOWN_ARROW)
    PressKey(K)
    time.sleep(0.1)
    ReleaseKey(K)
    # while True:
    #     station = cv2.resize(cv2.cvtColor(grab_screen(state_size), cv2.COLOR_RGBA2RGB),(1000,500))
    #     if station[187][612][0] > 200:
    #         # PressKey(DOWN_ARROW)
    #         # time.sleep(0.1)
    #         # ReleaseKey(DOWN_ARROW)
    #         PressKey(K)
    #         time.sleep(0.1)
    #         ReleaseKey(K)
    #         break
    #     else:
    #         Look_up()
    #         time.sleep(0.2)


# List for action functions
Actions = [Attack, Attack_Up,
           Short_Jump, Mid_Jump, Skill_Up,
           Skill_Down, Rush, Cure]
Directions = [Move_Left, Move_Right, Turn_Left, Turn_Right]


# Run the action
def take_action(action):
    Actions[action]()


def take_direction(direc):
    Directions[direc]()


class TackAction(threading.Thread):
    def __init__(self, threadID, name, direction, action):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.direction = direction
        self.action = action

    def run(self):
        take_direction(self.direction)
        take_action(self.action)
