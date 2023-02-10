import ctypes
import ctypes.wintypes
import os
import random
import time
# import subprocess
import cv2
import numpy as np
import win32api, win32gui, win32ui, win32con
from PIL import Image, ImageGrab


def get_window_rect(hwnd):
    try:
        f = ctypes.windll.dwmapi.DwmGetWindowAttribute
    except WindowsError:
        f = None
    if f:
        rect = ctypes.wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        f(ctypes.wintypes.HWND(hwnd),
          ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
          ctypes.byref(rect),
          ctypes.sizeof(rect)
          )
        return rect.left, rect.top, rect.right, rect.bottom


# 通过窗口句柄截取当前句柄图片 返回cv2格式的Mat数据
import os
import random
import time
# import subprocess
import cv2
import numpy as np
import win32api, win32gui, win32ui, win32con
from PIL import Image,ImageGrab


# 通过窗口句柄截取当前句柄图片 返回cv2格式的Mat数据
def window_capture(hwnd):
    x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)  # 获取当前窗口大小
    hwndDC = win32gui.GetWindowDC(hwnd)  # 通过应用窗口句柄获得窗口DC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)  # 通过hwndDC获得mfcDC(注意主窗口用的是win32gui库，操作位图截图是用win32ui库)
    neicunDC = mfcDC.CreateCompatibleDC()  # 创建兼容DC，实际在内存开辟空间（ 将位图BitBlt至屏幕缓冲区（内存），而不是将屏幕缓冲区替换成自己的位图。同时解决绘图闪烁等问题）
    savebitmap = win32ui.CreateBitmap()  # 创建位图
    width = x2 - x1
    height = y2 - y1
    savebitmap.CreateCompatibleBitmap(mfcDC, width, height)  # 设置位图的大小以及内容
    neicunDC.SelectObject(savebitmap)  # 将位图放置在兼容DC，即 将位图数据放置在刚开辟的内存里
    neicunDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)  # 截取位图部分，并将截图保存在剪贴板
    img_buf = savebitmap.GetBitmapBits(True)

    img = np.frombuffer(img_buf, dtype="uint8")
    img.shape = (height, width, 4)
    mat_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)  # 转换RGB顺序

    cv2.imshow('baofeng', img)
    cv2.waitKey()

    # 释放内存
    win32gui.DeleteObject(savebitmap.GetHandle())
    neicunDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)


    return img


if __name__ == '__main__':
    # 获取应用句柄
    hwnd = win32gui.FindWindow(None, 'Hollow Knight')


    img = window_capture(hwnd)
