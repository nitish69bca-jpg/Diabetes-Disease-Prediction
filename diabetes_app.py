import numpy as np 
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

model = pickle.load(open("diabetesfile.pkl","rb"))
scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))

img = Image.open(r"measuring-blood-glucose.jpg")
img = img.resize((200,200))
st.image(img,caption="Diabetes Image",width=200)

st.title('Diabetes Prediction')
input_data = st.text_input("Enter your data")

if st.button("submit"):
    try :    
        input_data_array = np.asarray(input_data.split(','), dtype=float)
        input_data_reshaped = input_data_array.reshape(1,-1)
        input_data_scaled = scaler.transform(input_data_reshaped)
        prediction = model.predict(input_data_scaled)

        if prediction[0]==1 :
            st.write("The person is Diabetic")
        else :
            st.write("The person is NOT Diabetic")    

    except:
        st.error("Please Enter exactly 8 numeric values separated by commas")