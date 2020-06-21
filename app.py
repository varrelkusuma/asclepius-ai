import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import imutils
import pickle
import cv2
import time
import uuid
import base64

#costum imports
from image_process.GrayLevelCooccurenceMatrix import GrayLevelCooccurenceMatrix
from image_process.LocalBinaryPattern import LocalBinaryPatterns
from image_process.ColorExtraction import ColorExtraction
from image_process.ImageProcess import ImageProcess
from image_process.ObjectRemoval import ObjectRemoval
from image_process.Segmentation import Segmentation

# Other variable
count = 1
alpha = 1.4
beta = 10

filename = 'svm_model.h5'
model = pickle.load(open(filename, 'rb'))

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(request.get(url).content)

def predict(file):
    # Process the picture
    ip = ImageProcess()
    obr = ObjectRemoval()
    sg = Segmentation()
    image = cv2.imread(file)
    remove = obr.removeHair(image)
    resize = ip.resize(remove, 50)
    color_correction = ip.manualColorCorrection(resize, alpha, beta)
    cropped = sg.cropRect(color_correction)

    # Crete needed picture
    thresh = obr.toThresh(cropped)
    thresh_gray = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    masked = ip.color_mask(cropped, thresh_gray)

    # Data extraction
    # GLCM
    glcm = GrayLevelCooccurenceMatrix()
    matrix_cooccurence = glcm.createMatrix(cropped)
    glcm_out = glcm.feature_extraction(matrix_cooccurence)
    #print(glcm_out.transpose())

    # LBP
    lbp = LocalBinaryPatterns(24, 8)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    hist = lbp.describe(gray)
    lbp_out = np.reshape(hist, (1,26))
    lbp_out_df = pd.DataFrame(np.concatenate(lbp_out), index=None)
    #print(lbp_out_df.shape)

    # Color
    ce = ColorExtraction()
    color = ce.color_extraction(masked)
    color_out = np.reshape(color, (1, 10))
    color_out_df = pd.DataFrame(np.concatenate(color_out), index=None)
    #print(color_out_df.shape)

    # Combine
    matrix = pd.concat([color_out_df, lbp_out_df, glcm_out.transpose()])
    matrix_transpose = matrix.transpose()
    
    result = model.predict(matrix_transpose)
    if result == 1:
        print("Label: Melanoma")
    elif result == 2:
	    print("Label: Basal Cell Carcinoma")
    elif result == 3:
	    print("Label: Squamous Cell Carcinoma")
    return result

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 1:
                label = 'Melanoma'
            elif result == 2:
                label = 'Basal Cell Carcinoma'			
            elif result == 3:
                label = 'Squamous Cell Carcinoma'
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)