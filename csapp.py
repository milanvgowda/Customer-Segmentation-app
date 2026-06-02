import streamlit as st
import numpy as np
import pickle

kmeans = pickle.load(open("cskmeans_model.pkl","rb"))
scaler = pickle.load(open("csscaler.pkl","rb"))

st.title("Customer Segmentation using K-Means")

st.write("Enter Customer Details")

annual_income = st.number_input("Annual Income", value=50000)
purchase_amount = st.number_input("Purchase Amount", value=300)
purchase_frequency = st.number_input("Purchase Frequency", value=15)
loyalty_score = st.number_input("Loyalty Score", value=5.0)

if st.button("Predict Cluster"):

    data = np.array([[annual_income, purchase_amount, purchase_frequency, loyalty_score]])
    data_scaled = scaler.transform(data)
    cluster = kmeans.predict(data_scaled)

    st.success(f"Customer belongs to Cluster: {cluster[0]}")

    if cluster[0] == 0:
        st.write("High-value customer (High income, high spending, frequent & loyal)")
    elif cluster[0] == 1:
        st.write("Low-value customer (Low income, low spending, less loyal)")
    else:
        st.write("Moderate customer (Less frequent but bulk buyer, decent loyalty)")