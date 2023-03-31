from brisque import BRISQUE
import pandas as pd
import cv2
# if you are running this code in Jupyter notebook
import os
from dom import DOM


def calculate_rms_contrast(image_location):
    """Calculate the contrast of an image
    Require the luminance (Y) in YUV color space
    """
    image = cv2.imread(image_location)
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # Luminance mean / std
    YUV_mean_std = cv2.meanStdDev(yuv_image)
    contrast = float(YUV_mean_std[1][0])
    return contrast

if __name__ == "__main__":
    ins_profile_data = pd.read_csv(f'/Users/sutong20000801/Downloads/test.csv')
    name = list(ins_profile_data['image_name'])
    for dirpath, dirnames, filenames in os.walk('/Users/sutong20000801/Downloads/image'):
        for filename in filenames:
            # Build image URL
            if filename in name and ins_profile_data.loc[ins_profile_data['image_name'] == filename, "Salience"].isna().any():
                print(filename)
                image_location = os.path.join(dirpath, filename)
                salience = calculate_rms_contrast(image_location)
                ins_profile_data.loc[ins_profile_data['image_name'] == filename, "Salience"] = salience
                ins_profile_data.to_csv('test.csv', index=False)
    print("done")

