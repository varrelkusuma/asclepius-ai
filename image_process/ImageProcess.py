import numpy as np
import cv2
import os

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

class ImageProcess:
    def __init__(self):
        print("Initializing Image Process...")

    def resize(self, image, percent):

        # Percent by which the image is resized
        scale_percent = percent

        # Calculate the 50 percent of original dimensions
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dsize = (width, height)

        # Resize Image
        output = cv2.resize(image, dsize)

        return output

    def cropRect(self, image, path, filename):
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            if cv2.contourArea(contour) < 200:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
            ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
            ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
            ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

            roi_corners = np.array([box], dtype=np.int32)
            cv2.polylines(gray, roi_corners, 1, (255, 0, 0), 3)
            result = img[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
            cv2.imwrite(os.path.join(path, filename), result)

    def yenThreshold(self, image):
        yen_threshold = threshold_yen(image)
        bright = rescale_intensity(image, (0, yen_threshold), (0, 255))

        return bright

    def manualColorCorrection(self, image, alpha, beta):
        alpha = alpha
        bete = beta

        manual_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return manual_result

    def color_mask(self, image, image_mask):
        masked = cv2.bitwise_and(image, image_mask)

        return masked