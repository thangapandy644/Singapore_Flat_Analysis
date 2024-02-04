import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
st.set_page_config(page_title="Singapore  Resale Flat Prices Predicting",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={"About":"This dashboard app is created by Priyanka Pal"})
st.markdown(f'<h1 style="text-align:center;">Singapore Resale Flat Prices Prediction App</h2>',
            unsafe_allow_html=True)
st.markdown("<style>div.block-container{padding-top:1rem;}</style>",unsafe_allow_html=True)

st.write("Welcome to our app, designed to empower both buyers and sellers in the Singapore housing market! With our application, potential buyers can effortlessly estimate resale prices and make well-informed decisions when it comes to purchasing a property. On the other hand, sellers can gain an accurate understanding of their flat's potential market value, aiding them in setting a competitive price.")
df=pd.read_csv(r"C:\Singapore_resale/singapore.csv")
town_option=df["town"].unique()
storey_option=df["storey_range"].unique()
model_option=df["flat_model"].unique()

with st.form("my_form"):
    col1,col2,col3=st.columns([2.5,0.5,2.5])
    with col1:
        year=st.text_input("Enter the Year")
        town=st.selectbox("Select the Town",town_option)
        storey_range=st.selectbox("Select the Storey Range",storey_option)
    with col3:
        floor_area_sqm=st.text_input("Enter the Floor Area (Square-Metre)")
        flat_model=st.selectbox("Select the Flat Model",model_option)
        lease_commence_date=st.text_input("Enter the Lease Commence Date in years")        
    submit_button = st.form_submit_button(label="PREDICT PRICE")
    flag = 0
    pattern = '^(?:\d+|\d*\.\d+)$'
    for i in [year, floor_area_sqm, lease_commence_date]:
        if re.match(pattern, i):
            pass
        else:
            flag = 1
            break

    if submit_button and flag == 1:
        if len(i) == 0 :
            st.write(':red[Please make sure to fill out all the fields mentioned.]')
        else:
            st.write(':red[You have entered an invalid value:]', i)

    if submit_button and flag==0:
        import pickle
        with open("C:\Singapore_resale\model.pkl","rb") as f:
            loaded_model=pickle.load(f)
        with open("C:\Singapore_resale/scaler.pkl","rb") as f:
            loaded_scaler=pickle.load(f)
        with open("C:\Singapore_resale/storey_encoder.pkl","rb") as f:
            loaded_oe=pickle.load(f)
        with open("C:\Singapore_resale/town_encoder.pkl","rb") as f:
            loaded_ohe1=pickle.load(f)
        with open("C:\Singapore_resale/flat_model_enc.pkl","rb") as f:
            loaded_ohe2=pickle.load(f)
        new_input = np.array([[storey_range,float(floor_area_sqm),int(lease_commence_date),int(year),town,flat_model]])
        ordinal_encoded = loaded_oe.fit_transform(new_input[:, [0]])
        new_sample_ohe1 = loaded_ohe1.transform(new_input[:, [4]]).toarray()
        new_sample_ohe2 = loaded_ohe2.transform(new_input[:, [5]]).toarray()
        new_input_concatenated = np.concatenate((ordinal_encoded, new_input[:, [1, 2, 3]], new_sample_ohe1, new_sample_ohe2), axis=1)
        new_sample1 = loaded_scaler.transform(new_input_concatenated)
        new_pred= loaded_model.predict(new_sample1)
        predicted_price= round(float(new_pred[0]), 4)
        formatted_price = "${:,.2f}".format(predicted_price)
        st.write('## :red[Predicted Flat Price:]', formatted_price)
