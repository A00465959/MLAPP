import streamlit as st
from joblib import load
from PIL import Image


st.title("Iris Flower - Species Predictor")
st.write("-----------------------------------")
LABELS = ['Setosa', 'Versicolor', 'Virginica']



clf = load("DT.joblib")

st.write("Please Enter The Below Details:")
st.write("\n\n")


sp_l = st.slider('SEPAL LENGTH (in cm):', min_value=0, max_value=10)

sp_w = st.slider('SEPAL WIDTH (in cm):', min_value=0, max_value=10)

pe_l = st.slider('PETAL LENGTH (in cm):', min_value=0, max_value=10)

pe_w = st.slider('PETAL WIDTH (in cm):', min_value=0, max_value=10)


prediction = clf.predict([[sp_l, sp_w, pe_l, pe_w]])
prediction = LABELS[prediction[0]]


setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')
no_image = Image.open('no_image.png')



if st.button("Click to Classify"):
	st.write("-----------------------------------")
	st.write("The Species of Iris Flower that has been identified is:")

	if prediction == 'Setosa':
		st.image(setosa)
	elif prediction == 'Versicolor':
		st.image(versicolor)
	elif prediction == 'Virginica':
		st.image(virginica)
	else:
		st.image(no_image)
	st.write("-----------------------------------")