import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('D:\supply chain\model.pkl', 'rb'))

# Streamlit app
st.title("Supply Chain Stock Level Predictor")

st.write("Enter the required features below to predict the stock level.")

# Input fields
features = {}
feature_names = [
    'Product type', 'Price', 'Availability', 'Number of products sold', 'Revenue generated',
    'Customer demographics', 'Lead times', 'Order quantities', 'Shipping times', 'Shipping carriers',
    'Shipping costs', 'Supplier name', 'Location', 'Lead time', 'Production volumes',
    'Manufacturing lead time', 'Manufacturing costs', 'Defect rates', 'Transportation modes', 'Routes',
    'Costs', 'Stock_Level_t-1', 'Stock_Level_t-2', 'Stock_Level_MA_3', 'Stock_Level_MA_7',
    'Lead_Time_Demand', 'Reorder_Point', 'Avg_Stock_Level', 'Stock_Turnover_Ratio', 'COGS_per_day', 'DIO'
]

for feature in feature_names:
    features[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

# Predict button
if st.button("Predict Stock Level"):
    try:
        input_features = np.array([features[feature] for feature in feature_names]).reshape(1, -1)
        prediction = model.predict(input_features)[0]
        st.success(f"Predicted Stock Level: {prediction}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
