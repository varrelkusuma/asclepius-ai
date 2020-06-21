import numpy as np
import pandas as pd
import cv2

# Class Definition
class ColorExtraction:

    def __init__(self):
        print("Initializing Color Extraction...")

    def color_extraction(self, image):
        
        # Attention: This algorithm processed RGB color space
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        h, w = np.shape(gray)
        x = range(w)
        y = range(h)

        # Calculate projections along the x and y axes
        xp = np.sum(gray, axis = 0)
        yp = np.sum(gray, axis = 1)

        # Get Image Centroid
        cx = np.sum(x * xp) / np.sum(xp)
        cy = np.sum(y * yp) / np.sum(yp)

        # Standard Deviation
        x2 = (x-cx) ** 2
        y2 = (y-cy) ** 2

        sx = np.sqrt(np.sum(x2*xp) / np.sum(xp))
        sy = np.sqrt(np.sum(y2*yp) / np.sum(yp))

        # Skewness
        x3 = (x - cx) ** 3
        y3 = (y - cy) ** 3

        skx = np.sum(xp * x3) / (np.sum(xp) * sx ** 3)
        sky = np.sum(yp * y3) / (np.sum(yp) * sy ** 3)

        # Kurtosis
        x4 = (x-cx) ** 4
        y4 = (y-cy) ** 4

        kx = np.sum(xp * x4) / (np.sum(xp) * sx ** 4)
        ky = np.sum(yp * y4) / (np.sum(yp) * sy ** 4)

        mean, stdev = cv2.meanStdDev(image)
        other = [skx, sky, kx, ky]
        meanstd = np.concatenate([mean, stdev]).flatten()
        stats = np.concatenate([meanstd, other])

        return stats
