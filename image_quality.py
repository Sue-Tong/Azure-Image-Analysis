from brisque import BRISQUE
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# if you are running this code in Jupyter notebook
import os
import glob
import math
from collections import Counter
from multiprocessing import Process
from time import sleep
from random import random
import random
import warnings
from dom import DOM
import imquality.brisque as brisque
import PIL.Image
#from brisque import BRISQUE
import parmap
import multiprocessing
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join
import shutil
import time

import PIL.Image
import skimage.transform
import imquality.brisque as brisque
def calculate_sharpness(image_location):
    '''Calculate the sharpness of the image'''
    iqa = DOM()
    
    sharpness_score = iqa.get_sharpness(image_location)
    return sharpness_score

def calculate_BRISQUE(image_location):
    '''Calculate the BRISQUE score of the image using another package'''
    img = cv2.imread(image_location)
    obj = BRISQUE(url = False)
    brisque_score_alt = obj.score(img)
    return brisque_score_alt

if __name__ == "__main__":
    ins_profile_data = pd.read_csv(f'/Users/sutong20000801/Downloads/new.csv')
    name = list(ins_profile_data['image_name'])
    for dirpath, dirnames, filenames in os.walk('/Users/sutong20000801/Downloads/image'):
        for filename in filenames:
            # Build image URL
            if filename in name and ins_profile_data.loc[ins_profile_data['image_name'] == filename, "brisque"].isna().any():
                print(filename)
                image_location = os.path.join(dirpath, filename)
                sharpness = calculate_sharpness(image_location)
                brisques = calculate_BRISQUE(image_location)
                ins_profile_data.loc[ins_profile_data['image_name'] == filename, "brisque"] = brisques
                ins_profile_data.loc[ins_profile_data['image_name'] == filename, "sharpness"] = sharpness
                ins_profile_data.to_csv('new.csv', index=False)
    print("done")
    ins_profile_data.to_csv('new.csv')

