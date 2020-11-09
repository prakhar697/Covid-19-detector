import sys
import os
import glob
import re
import numpy as np
import cv2

import tensorflow
from tensorflow.keras.models import load_model

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_path = 'model.h5'  # your created model
IMG_SIZE = 224  # size of the images that are used in model

def model_predict(img_path,model):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)   # reading image from the given path in grayscale 
    
    new_array = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  # resizing the images into size of images that are being used to create the model
    
    x= np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE,1)  # reshaping so that it can predicted from preidct function
    
    x = x/255.0  # normalizing all the pixels in the image array
    
    preds  = model.predict(x)
    return preds

model = load_model(model_path)
model._make_predict_function()

@app.route('/', methods = ['GET'])

def index():
    
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    
    if request.method == 'POST':
        
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath,'upload',secure_filename(f.filename))  # path from which images are being uploaded
        
        f.save(file_path)
        
        preds = model_predict(file_path,model)
        
        pred_class = ['NEGATIVE','POSITIVE']
        
        if float(preds[0,0])< 0.5 :
            
            result = str(pred_class[0])
        else :
            result = str(pred_class[1])
        
        return result
    return None

if __name__ == '__main__':
    app.run()