# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:27:30 2021

@author: Naman
"""

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np

model=load_model('project.h5')
app = Flask(__name__)


@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/success',methods=['POST'])
def success():
    if request.method == 'POST':
    
        a = request.files['fileToUpload']
        a.save(a.filename)
        if a.filename.endswith('jpg') or a.filename.endswith('png') or a.filename.endswith('jpeg'): 
    
            name=image.load_img(a.filename, target_size=(64,64))
            x=image.img_to_array(name)
    
            
            x=np.expand_dims(x,axis=0)
            pred=model.predict_classes(x)
            index=['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction','Premature Ventricular Contractions','Right Bundle Branch Block','Ventricular Fibrillation']
            answer=index[pred[0]]
            
            
            return render_template('index.html', prediction_text='Type {}'.format(answer))
        else:
            return render_template('index.html', prediction_text='Type {}'.format('Error Use image folder'))
        
        

if __name__ == "__main__":
    #app.debug = True

    app.run()
