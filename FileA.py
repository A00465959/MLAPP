import streamlit as st
from joblib import load
from skimage import io
from sklearn.datasets import load_iris
from PIL import Image


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


setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')



if st.button("Click Here to Classify"):
	st.write("The Species of Iris Flower that has been identified is:") 
	st.write(prediction)
	st.image(setosa) if prediction == 0 else st.image(versicolor)  if prediction == 1 else st.image(virginica) 
	st.write("-----------------------------------")
		
