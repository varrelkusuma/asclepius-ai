import numpy as np
import cv2
import os

class Segmentation:

    def __init__(self):
        print("Initializing segmentation...")
    
    def lesionSegmentationThresh(self, image):

        # Color Convert
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (17, 17), 32)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        return thresh
        

    def drawContour(self, image):
        # Create Box & Test
        img = image
        blur = cv2.GaussianBlur(gray, (17, 17), 32)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
        cv2.putText(img, 'skin_lesion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)

        cv2.imshow('frame', img)


    def cropRect(self, image):
        gray = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (17, 17), 32)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)
        crop_img = image[y:y+h, x:x+w]

        return crop_img


