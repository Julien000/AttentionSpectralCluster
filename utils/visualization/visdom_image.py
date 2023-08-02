import time
import cv2
import visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform

def ImagetoVisdom(
    img,
    env="image_env",
    title='Random!',
    caption='Click me!',
    jpgquality=100

    ):
    '''
        Image格式:C,H,W 才行
    '''
    viz = visdom.Visdom(env=env)
    viz.image(
        img,
        opts={'title': title, 'caption':caption, "jpgquality":jpgquality }
    )

def matplotToVisdom(plt):
    '''
    输出对应输出的图片
    '''
    
    viz = visdom.Visdom()
    assert viz.check_connection()
    try:
        viz.matplot(plt)
    except BaseException as err:
        print('Skipped matplotlib example')
        print('Error message: ', err)