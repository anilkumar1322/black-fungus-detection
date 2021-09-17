# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:16:32 2021

@author: anil
"""

	
	
#app.py

import pandas as pd 
import numpy as np 

from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
 
app = Flask(__name__)


                         
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Model saved with Keras model.save()
MODEL_PATH ='transfor_learning_model'

# Load your trained model
model = load_model(MODEL_PATH)

class_names=['Fungus', 'Normal']


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)    
    predictions = model.predict(img_preprocessed)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])),2)
    return predicted_class,confidence
 
img_path1='D:\data science projects\Black fungus project\Images\C_Normal.jpg'
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        

        filename = secure_filename(file.filename)
        file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(file_path)
        
        preds = model_predict(file_path, model)
        print(preds)
        result=preds[0]
        confidence=preds[1]
    
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        
        return render_template('index.html', filename=filename,result=result,confidence=confidence)
    
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()