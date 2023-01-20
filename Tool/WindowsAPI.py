import numpy as np
import win32api
import win32con
import win32gui
import win32ui

# get hollow knight hwnd
hwnd = win32gui.FindWindow(None, 'Hollow Knight')


# get windows image of hollow knight
def grab_screen(region=None):
    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


# check which key is pressed
def key_check():
    operations = []
    if win32api.GetAsyncKeyState(0x4E):
        operations.append("N")
    if win32api.GetAsyncKeyState(0x4C):
        operations.append("L")
    if win32api.GetAsyncKeyState(0x4A):
        operations.append("J")
    if win32api.GetAsyncKeyState(0x4B):
        operations.append("K")
    if win32api.GetAsyncKeyState(0x70):
        operations.append("T")

    direction = []
    if win32api.GetAsyncKeyState(0x41):
        direction.append("A")
    if win32api.GetAsyncKeyState(0x57):
        direction.append("W")
    if win32api.GetAsyncKeyState(0x44):
        direction.append("D")
    if win32api.GetAsyncKeyState(0x53):
        direction.append("S")

    return operations, direction


# win32 presskey and releasekey, but it has lag, what we need is in SendKey.py
def PressKey(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)


def ReleaseKey(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, win32con.KEYEVENTF_KEYUP, 0)
