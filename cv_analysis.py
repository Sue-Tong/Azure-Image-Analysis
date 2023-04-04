from brisque import BRISQUE
import pandas as pd
import cv2
# if you are running this code in Jupyter notebook
import os
from dom import DOM


def calculate_sharpness(image_location):
    '''Calculate the sharpness of the image'''
    iqa = DOM()
    
    sharpness_score = iqa.get_sharpness(image_location)
    return sharpness_score

def calculate_BRISQUE(image):
    '''Calculate the BRISQUE score of the image using another package'''

    obj = BRISQUE(url = False)
    brisque_score_alt = obj.score(image)
    return brisque_score_alt

def calculate_rms_contrast(image):
    """Calculate the contrast of an image
    Require the luminance (Y) in YUV color space
    """
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # Luminance mean / std
    YUV_mean_std = cv2.meanStdDev(yuv_image)
    contrast = float(YUV_mean_std[1][0])
    return contrast

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
                image = cv2.imread(image_location)
                brisques = calculate_BRISQUE(image)
                salience = calculate_rms_contrast(image)
                ins_profile_data.loc[ins_profile_data['image_name'] == filename, "brisque"] = brisques
                ins_profile_data.loc[ins_profile_data['image_name'] == filename, "sharpness"] = sharpness
                ins_profile_data.loc[ins_profile_data['image_name'] == filename, "Salience"] = salience
                ins_profile_data.to_csv('new.csv', index=False)
    print("done")
    ins_profile_data.to_csv('new.csv')
