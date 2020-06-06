import numpy as np
import pandas as pd
from skimage import feature
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte, img_as_float

# Class Definition
class GrayLevelCooccurenceMatrix:

	def __init__(self):
		print("Initializing Gray Level Coocurrence Matrix...")

	def feature_extraction(self, matrix_coocurrence):
		contrast = greycoprops(matrix_coocurrence, 'contrast')
		dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
		homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
		energy = greycoprops(matrix_coocurrence, 'energy')
		correlation = greycoprops(matrix_coocurrence, 'correlation')
		asm = greycoprops(matrix_coocurrence, 'ASM')

		contrast_df = pd.DataFrame(np.concatenate([contrast]))
		dissimilarity_df = pd.DataFrame(np.concatenate([dissimilarity]))
		homogeneity_df = pd.DataFrame(np.concatenate([homogeneity]))
		energy_df = pd.DataFrame(np.concatenate([energy]))
		correlation_df = pd.DataFrame(np.concatenate([correlation]))
		asm_df = pd.DataFrame(np.concatenate([asm]))

		result = pd.concat([contrast_df, dissimilarity_df, homogeneity_df, energy_df, correlation_df, asm_df], axis = 1)

		return result

	def contrast_feature(self, matrix_coocurrence):
		contrast = greycoprops(matrix_coocurrence, 'contrast')
		return contrast

	def dissimilarity_feature(self, matrix_coocurrence):
		dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
		return dissimilarity

	def homogeneity_feature(self, matrix_coocurrence):
		homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
		return homogeneity

	def energy_feature(self, matrix_coocurrence):
		energy = greycoprops(matrix_coocurrence, 'energy')
		return energy

	def correlation_feature(self, matrix_coocurrence):
		correlation = greycoprops(matrix_coocurrence, 'correlation')
		return correlation

	def asm_feature(self, matrix_coocurrence):
		asm = greycoprops(matrix_coocurrence, 'ASM')
		return asm

	def createMatrix(self, image):
		gray = color.rgb2gray(image)
		image = img_as_ubyte(gray)
		bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
		inds = np.digitize(image, bins)
		max_value = inds.max()+1
		matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
		
		return matrix_coocurrence