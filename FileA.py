import streamlit as st
from joblib import load
from skimage import io
from sklearn.datasets import load_iris

st.title("Iris Flower Species Predictor")
st.write("-----------------------------------")
LABELS = ['Setosa', 'Versicolor', 'Virginica']

clf = load("DT.joblib")

sp_l = st.slider('sepal length (cm)', min_value=0, max_value=10)

sp_w = st.slider('sepal width (cm)', min_value=0, max_value=10)

pe_l = st.slider('petal length (cm)', min_value=0, max_value=10)

pe_w = st.slider('petal width (cm)', min_value=0, max_value=10)


prediction = clf.predict([[sp_l, sp_w, pe_l, pe_w]])
prediction = LABELS[prediction[0]]

iris = load_iris()
label_index = LABELS.index(prediction) 
images = iris.images[iris.target == label_index]

st.write("The Species of Iris Flower that has been identified is:") 
st.write(prediction)
io.imshow(images[0])
io.show()
st.write("-----------------------------------")
		
