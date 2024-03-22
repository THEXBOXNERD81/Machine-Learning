from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
import cv2
import pickle
import numpy as np
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

@st.cache(allow_output_mutation=True)

def load_model():
    mnist = fetch_openml('mnist_784', version = 1, cache = True,  as_frame = False)
    X = mnist['data']
    y = mnist['target'].astype(np.uint8)
    X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]
    svc_clf = SVC(C=5.0, kernel='rbf', gamma='scale', random_state=42)
    svc_clf.fit(X_train, y_train)

    return svc_clf
model=load_model()

uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg'])

def predict(image, model):

    image = cv2.resize(image, (784, 1))
    print(image)
    
    y = model.predict(image)

    return y

if uploaded_file == None:
    st.text('Please Upload an image file')
else:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, use_column_width=True)
    predictions = predict(image, model)
    string = "This is the number: " + f'{predictions}'
    st.success(string)


