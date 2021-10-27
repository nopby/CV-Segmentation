

import cv2
import numpy as np


def GetContourImage(image, color):

    # Grayscaling
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create background with gray image shape
    background = np.zeros(gray.shape)

    # Find contours from image gray
    ctrs, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours to background
    imageCtrs = cv2.drawContours(background, ctrs, -1, color, 1)

    # Return image contours
    return imageCtrs
