from __future__ import print_function
from flask import Flask,jsonify,request,render_template

import numpy as np
import keras
from keras.models import model_from_json



app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def landing():
    return render_template("index.html")

@app.route('/hook', methods=['POST'])
def get_prediction():
    
    
    input_img = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1,28, 28,1)
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    input_img = input_img.astype('float32')
    
    y_pred = loaded_model.predict(input_img)
    predicted_int = np.argmax(y_pred)
    
    return jsonify(results=predicted_int)
    
    


if __name__ == "__main__":
    app.run(debug=True)

