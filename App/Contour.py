

import cv2
import numpy as np
def GetContourImage(image, color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = np.zeros(gray.shape)

    ctrs, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    imageCtrs = cv2.drawContours(background, ctrs, -1, color, 1)

    return imageCtrs
