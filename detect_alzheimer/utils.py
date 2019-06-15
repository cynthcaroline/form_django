from keras.models import load_model
import tensorflow as tf

import numpy as np

#initializing the graph
graph = tf.get_default_graph()

#loading our trained model
print("Keras model loading.......")
model = load_model('detect_alzheimer/_vgg16_.52-0.93.hdf5', compile=False)
print("Model loaded!!")

def predict(img):
    x = (img/225.)
    with graph.as_default():
        prediction = model.predict(np.expand_dims(x, axis=0))[0]

    return prediction