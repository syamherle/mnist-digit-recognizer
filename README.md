# mnist-digit-recognizer

### Web application deployed on Heroku

https://mnist-digit-recognizer.herokuapp.com/

### Following files were used for implementation of the application

 + Flask
 + keras
 + TensorFlow
 + Numpy
 + h5py
 + gunicorn
 
 ### Description
 
 + Convolution Neural Network (CNN) model wase trained using Keras on MNIST digit handwritten dataset
 + Following are the configuration of the trained CNN model,
   + First Convolution layer has 3\*3 kernal filter and has 32 activation output, and ReLu as activation function.
   + Second Convolution layer has 3\*3 kernal filter and has 64 activation output, and ReLu as activation function.
   + Pooling layer, which has pooling window 2\*2.
   + Dropout of probability = 0.25.
   + Dense layer with 128 hidden units with ReLu as activation function and dropout of probability = 0.25.
   + Dense layer with 10 hidden units with SoftMax as activation function and dropout of probability = 0.25.
