import cv2
import os
import numpy as np

class ObjectRemoval:

    def __init__(self):
        print("Initializing object removal...")
    
    def removeHair(self, image):
        # Convert to grayscale
        img = image
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Create kernel & perform blackhat filtering
        kernel = cv2.getStructuringElement(1,(17,17))
        blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

        # Create contours & inpaint
        ret, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)

        return result

    def toThresh(self, image):
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (17, 17), 32)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        return thresh

    def save(self, image, path, filename):
        cv2.imwrite(os.path.join(path, filename), image)