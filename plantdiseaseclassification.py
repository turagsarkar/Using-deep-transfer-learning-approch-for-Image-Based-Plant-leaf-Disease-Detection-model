import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing

st.header('Leaf Disease Predictor')

def main():
    file_uploaded = st.file_uploader('Choose the file', type=['jpg', 'png'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model = tf.keras.models.load_model('/content/drive/MyDrive/LEAF/plant_classifier.h5')
    shape = (256, 256, 3)
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
    test_image = image.resize((256, 256))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Healthy', 'Powdery', 'Rust']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if __name__ == '__main__':
    main()
