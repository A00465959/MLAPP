import streamlit as st
from joblib import load

st.title("Iris Flower Species Predictor")
LABELS = ['setosa', 'versicolor', 'virginica']

clf = load("DT.joblib")

sp_l = st.slider('sepal length (cm)', min_value=0, max_value=10)

sp_w = st.slider('sepal width (cm)', min_value=0, max_value=10)

pe_l = st.slider('petal length (cm)', min_value=0, max_value=10)

pe_w = st.slider('petal width (cm)', min_value=0, max_value=10)


prediction = clf.predict([[sp_l, sp_w, pe_l, pe_w]])
prediction = str(LABELS[prediction[0]]).upper()
st.write("The Species of Iris Flower that has been identified is:" + prediction)


prediction_list = []
prediction_list.append(prediction)
st.write("Predicton History")
st.write("------------------")
for index, prediction in prediction_list:
	st.write((index+1)+". " + prediction)
		
