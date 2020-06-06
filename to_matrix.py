# Built-in imports
import sys
import os
import json
import uuid
from time import time

# third-party imports
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte, img_as_float

#costum imports
from image_process.GrayLevelCooccurenceMatrix import GrayLevelCooccurenceMatrix
from image_process.LocalBinaryPattern import LocalBinaryPatterns
from image_process.ColorExtraction import ColorExtraction
from image_process.ImageProcess import ImageProcess
from image_process.ObjectRemoval import ObjectRemoval
from image_process.Segmentation import Segmentation

"""
# Raw Picture Folder
melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\melanoma'
bcc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\bcc'
scc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\scc'

# Resize Picture Folder
mel_resize = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\melanoma'
bcc_resize = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\bcc'
scc_resize = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\scc'

# Threshold Picture Folder
mel_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\melanoma'
bcc_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\bcc'
scc_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\scc'

# Masked Picture Folder
mel_masked = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\masked\melanoma'
bcc_masked = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\masked\bcc'
scc_masked = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\masked\scc'
"""

# Other variable
count = 1
alpha = 1.4
beta = 10

"""
==========================================================================================
Image Process Initialization
This method is special for melanoma case because the raw image pixel are around 2000x1000
The other (BCC & SCC) doesn't need resize and could be used as it is
==========================================================================================
1. Object removal (hair)
2. Image resize (60% size)
3. Image cropped to be close to the bounding box

"""

# Create Object
ip = ImageProcess()
obr = ObjectRemoval()
sg = Segmentation()

print("Pre-processing Picture...")
for filename in os.listdir(melanoma):
	if filename.endswith(".jpg"):

		# Resize & Crop Image
		tempfilename = melanoma+"/"+filename
		image = cv2.imread(tempfilename)
		remove = obr.removeHair(image)
		resize = ip.resize(remove, 75)
		color_correction = ip.manualColorCorrection(resize, alpha, beta)
		cropped = sg.cropRect(color_correction)

		if cropped.size < 1:
			count = count + 1
		else:
			thresh = obr.toThresh(cropped)
			thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
			masked = ip.color_mask(cropped, thresh)
			filename = "file_%d.jpg"%count
			obr.save(thresh, mel_thresh, filename)
			obr.save(cropped, mel_resize, filename)
			obr.save(masked, mel_masked, filename)
			count = count + 1

"""
==========================================================================================

Feature Extraction for Thresholded Image

1. GLCM
2. Local Binary Pattern

==========================================================================================
"""
"""
==========================================================================================

Gray-Level Cooccurence Matrix (GLCM) Implementation

Steps done in the process:
1. Create empty matrix to contain all data extracted using GLCM
2. Loop every images in folder defined for this project (Melanoma, BCC, SCC)
3. Append all the data in defined matrix (6 elements from GLCM)
4. Create class series to contain image and data identifier
5. Create supervector for all this data

==========================================================================================
"""

mel_class = []
mel_glcm = []
mel_result = []
bcc_class = []
bcc_glcm = []
bcc_result = []
scc_class = []
scc_glcm = []
scc_result = []

# Create object
glcm = GrayLevelCooccurenceMatrix()

# Melanoma
print("Extracting GLCM from Melanoma...")
for filename in os.listdir(mel_resize):
	if filename.endswith(".jpg"):
		tempfilename = mel_resize+"/"+filename
		img = cv2.imread(tempfilename)
		matrix_cooccurence = glcm.createMatrix(img)
		result = glcm.feature_extraction(matrix_cooccurence)
		class_value = 1
		mel_glcm.append(result)
		mel_class.append(class_value)
# Create Dataframe & Concatenate GLCM Features & Class
mel_glcm_df = pd.DataFrame(np.concatenate(mel_glcm))
mel_class_series = pd.Series(mel_class)
mel_result = pd.concat([mel_glcm_df, mel_class_series], axis = 1)

# Basal Cell Carcinoma
print("Extracting GLCM from BCC...")
for filename in os.listdir(bcc_resize):
	if filename.endswith(".jpg"):
		tempfilename = bcc_resize+"/"+filename
		img = cv2.imread(tempfilename)
		matrix_cooccurence = glcm.createMatrix(img)
		result = glcm.feature_extraction(matrix_cooccurence)
		class_value = 2
		bcc_glcm.append(result)
		bcc_class.append(class_value)
# Create Dataframe & Concatenate GLCM Features & Class
bcc_glcm_df = pd.DataFrame(np.concatenate(bcc_glcm))
bcc_class_series = pd.Series(bcc_class)
bcc_result = pd.concat([bcc_glcm_df, bcc_class_series], axis = 1)

# Squamous Cell Carcinoma
print("Extracting GLCM from SCC...")
for filename in os.listdir(scc_resize):
	if filename.endswith(".jpg"):
		tempfilename = scc_resize+"/"+filename
		img = cv2.imread(tempfilename)
		matrix_cooccurence = glcm.createMatrix(img)
		result = glcm.feature_extraction(matrix_cooccurence)
		class_value = 3
		scc_glcm.append(result)
		scc_class.append(class_value)
# Create Dataframe & Concatenate GLCM Features & Class
scc_glcm_df = pd.DataFrame(np.concatenate(scc_glcm))
scc_class_series = pd.Series(scc_class)
scc_result = pd.concat([scc_glcm_df, scc_class_series], axis = 1)

"""
==========================================================================================

Local Binary Pattern (LBP) Implementation

Steps done in the process:
1. Create empty matrix to contain all data extracted using LBP
2. Loop every images in folder defined for this project (Melanoma, BCC, SCC)
3. Append all data (histogram) in the defined matrix
4. Create supervector for all the data

==========================================================================================
"""

# Create Object & Variable
lbp = LocalBinaryPatterns(24, 8)
mel_data = []
bcc_data = []
scc_data = []
lbp_out = []

# Melanoma
print("Extracting LBP from Melanoma...")
for filename in os.listdir(mel_resize):
	if filename.endswith(".jpg"):
		tempfilename = mel_resize+"/"+filename
		img = cv2.imread(tempfilename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		hist = lbp.describe(gray)
		reshaped = np.reshape(hist, (1,26))
		mel_data.append(reshaped)

# Basal Cell Carcinoma
print("Extracting LBP from BCC...")
for filename in os.listdir(bcc_resize):
	if filename.endswith(".jpg"):
		tempfilename = bcc_resize+"/"+filename
		img = cv2.imread(tempfilename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		hist = lbp.describe(gray)
		reshaped = np.reshape(hist, (1,26))
		bcc_data.append(reshaped)

# Squamous Cell Carcinoma
print("Extracting LBP from SCC...")
for filename in os.listdir(scc_resize):
	if filename.endswith(".jpg"):
		tempfilename = scc_resize+"/"+filename
		img = cv2.imread(tempfilename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		hist = lbp.describe(gray)
		reshaped = np.reshape(hist, (1,26))
		scc_data.append(reshaped)

mel_lbp_df = pd.DataFrame(np.concatenate(mel_data))
bcc_lbp_df = pd.DataFrame(np.concatenate(bcc_data))
scc_lbp_df = pd.DataFrame(np.concatenate(scc_data))

"""
==========================================================================================

Color Feature Extraction

Steps done in the process:
1. Create empty array for the data
2. Take global stdev and mean from cropped data (not threshold)
3. Use this data as a global color variable

==========================================================================================
"""

# Create Variable
ce = ColorExtraction()
mel_color = []
bcc_color = []
scc_color = []

print("Extracting Color Variables from Melanoma...")
for filename in os.listdir(mel_masked):
	if filename.endswith(".jpg"):
		tempfilename = mel_masked+"/"+filename
		img = cv2.imread(tempfilename)
		color = ce.color_extraction(img)
		color_reshaped = np.reshape(color, (1, 12))
		mel_color.append(color_reshaped)

print("Extracting Color Variables from BCC...")
for filename in os.listdir(bcc_masked):
	if filename.endswith(".jpg"):
		tempfilename = bcc_masked+"/"+filename
		img = cv2.imread(tempfilename)
		color = ce.color_extraction(img)
		color_reshaped = np.reshape(color, (1, 12))
		bcc_color.append(color_reshaped)

print("Extracting Color Variables from SCC...")
for filename in os.listdir(scc_masked):
	if filename.endswith(".jpg"):
		tempfilename = scc_masked+"/"+filename
		img = cv2.imread(tempfilename)
		color = ce.color_extraction(img)
		color_reshaped = np.reshape(color, (1, 12))
		scc_color.append(color_reshaped)

mel_color_df = pd.DataFrame(np.concatenate(mel_color))
bcc_color_df = pd.DataFrame(np.concatenate(bcc_color))
scc_color_df = pd.DataFrame(np.concatenate(scc_color))


"""
==========================================================================================

Exporting all the defined data (GLCM, LBP, Color) to a single matrix
The order of the supervector is as defined
1. (12 column) Color
2. (26 column) LBP
3. (24 column) GLCM

==========================================================================================
"""

# Exporting as csv file
glcm_out = pd.concat([mel_result, bcc_result, scc_result])
lbp_out = pd.concat([mel_lbp_df, bcc_lbp_df, scc_lbp_df])
color_out = pd.concat([mel_color_df, bcc_color_df, scc_color_df])
# glcm_out.drop(glcm_out.columns[0], axis=1)

glcm_out.reset_index(drop=True, inplace=True)
lbp_out.reset_index(drop=True, inplace=True)
color_out.reset_index(drop=True, inplace=True)
out = pd.concat([color_out, lbp_out, glcm_out], axis = 1)

glcm_out.to_csv('glcm.csv', index=False, header=None)
lbp_out.to_csv('lbp.csv', index=False, header=None)
color_out.to_csv('color.csv', index=False, header=None)
out.to_csv('data.csv', index=False, header=None)

cv2.waitKey(0)
cv2.destroyAllWindows()