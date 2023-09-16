import streamlit as st
import pandas as pd
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Load your model and dataset here
try:
    with open('model_pickle/deep_ann_model_architecture.json', 'r') as model_architecture_file:
        model_json = model_architecture_file.read()
        loaded_model = model_from_json(model_json)

    # Load the model weights
    loaded_model.load_weights('Notebooks/deep_ann_model_weights.h5')
except FileNotFoundError:
    st.error("Model file not found. Please upload the model file.")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define a Streamlit app
st.title('Machine Learning Prediction')

# Define input fields for prediction
customer_care_calls = st.number_input('Customer Care Calls', min_value=0)
customer_rating = st.number_input('Customer Rating', min_value=1, max_value=5)
cost_of_the_product = st.number_input('Cost of the Product', min_value=0.0, step=0.01)
prior_purchases = st.number_input('Prior Purchases', min_value=0)
product_importance = st.selectbox('Product Importance', ['low', 'medium', 'high'])
gender = st.selectbox('Gender', ['Male', 'Female'])
discount_offered = st.number_input('Discount Offered', min_value=0.0, step=0.01)
weight_in_gms = st.number_input('Weight in grams', min_value=0.0, step=0.01)
warehouse_block = st.selectbox('Warehouse Block', ['A', 'B', 'C', 'D', 'F'])
mode_of_shipment = st.selectbox('Mode of Shipment', ['Flight', 'Road', 'Ship'])

# Get one-hot encoded values based on user selection
gender_encoded = [1 if gender == 'Male' else 0]
product_imp_mapping = {'low' : 1, 'medium' : 2, 'high' : 3}
warehouse_mapping = {'A': [1, 0, 0, 0, 0],
                     'B': [0, 1, 0, 0, 0],
                     'C': [0, 0, 1, 0, 0],
                     'D': [0, 0, 0, 1, 0],
                     'F': [0, 0, 0, 0, 1]}
warehouse_encoded = warehouse_mapping.get(warehouse_block, [0, 0, 0, 0, 0])
shipment_mapping = {'Flight': [1, 0, 0], 'Road': [0, 1, 0], 'Ship': [0, 0, 1]}
shipment_encoded = shipment_mapping.get(mode_of_shipment, [0, 0, 0])

# Predict when the user clicks the "Predict" button
if st.button('Predict'):
    if (
        customer_care_calls < 0
        or customer_rating < 1
        or customer_rating > 5
        or cost_of_the_product <= 0.0
        or prior_purchases < 0
        or product_importance not in ['low', 'medium', 'high']
        or discount_offered < 0.0
        or weight_in_gms <= 0.0
    ):
        st.warning("Please fill in all required fields.")
    else:
        input_data = pd.DataFrame({
            'Customer_care_calls': [customer_care_calls],
            'Customer_rating': [customer_rating],
            'Cost_of_the_Product': [cost_of_the_product],
            'Prior_purchases': [prior_purchases],
            'Product_importance': [product_imp_mapping[product_importance]],
            'Gender': gender_encoded,
            'Discount_offered': [discount_offered],
            'Weight_in_gms': [weight_in_gms],
            'Warehouse_block_A': [warehouse_encoded[0]],
            'Warehouse_block_B': [warehouse_encoded[1]],
            'Warehouse_block_C': [warehouse_encoded[2]],
            'Warehouse_block_D': [warehouse_encoded[3]],
            'Warehouse_block_F': [warehouse_encoded[4]],
            'Mode_of_Shipment_Flight': [shipment_encoded[0]],
            'Mode_of_Shipment_Road': [shipment_encoded[1]],
            'Mode_of_Shipment_Ship': [shipment_encoded[2]]
        })

        try:
            # Use the loaded model to make predictions
            predictions = loaded_model.predict(input_data)
            predictions = (predictions > 0.5).astype(int)[0]

            if predictions[0] == 1:
                st.write('Product Reached With Delay')
            else:
                st.write('Product Reached on Time')
        except ValueError as e:
            # Handle prediction-related errors gracefully and display an error message
            st.error(f"Error making predictions: {str(e)}")
