from flask import Flask,render_template,request,redirect, jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import joblib



app=Flask(__name__)
cors=CORS(app)
car=pd.read_csv('Encoded.csv')

try:
    pipeline = joblib.load('SVM_pipeline_new.pkl')  # Replace with the correct pickle file path
except Exception as e:
    print(f"Error loading the pipeline: {str(e)}")

@app.route('/',methods=['GET','POST'])
def index():
    selected_cols = [
        'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
        'Prior_purchases', 'Product_importance', 'Gender', 'Discount_offered',
        'Weight_in_gms', 'Warehouse_block_A', 'Warehouse_block_B',
        'Warehouse_block_C', 'Warehouse_block_D', 'Mode_of_Shipment_Flight',
        'Mode_of_Shipment_Road'
    ]

    data = {}  # A dictionary to store the lists for each column

    for col in selected_cols:
        data[col] = sorted(car[col].unique())

    return render_template('index.html', data=data)




@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get the input data from the request (assuming it's form data)
        customer_care_calls = int(request.form['customer_care_calls'])
        customer_rating = int(request.form['customer_rating'])
        cost_of_the_product = float(request.form['cost_of_the_product'])
        prior_purchases = int(request.form['prior_purchases'])
        product_importance = int(request.form['product_importance'])
        gender = int(request.form['gender'])
        discount_offered = float(request.form['discount_offered'])
        weight_in_gms = float(request.form['weight_in_gms'])
        warehouse_block_A = int(request.form['warehouse_block_A'])
        warehouse_block_B = int(request.form['warehouse_block_B'])
        warehouse_block_C = int(request.form['warehouse_block_C'])
        warehouse_block_D = int(request.form['warehouse_block_D'])
        mode_of_shipment_flight = int(request.form['mode_of_shipment_flight'])
        mode_of_shipment_road = int(request.form['mode_of_shipment_road'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Customer_care_calls': [customer_care_calls],
            'Customer_rating': [customer_rating],
            'Cost_of_the_Product': [cost_of_the_product],
            'Prior_purchases': [prior_purchases],
            'Product_importance': [product_importance],
            'Gender': [gender],
            'Discount_offered': [discount_offered],
            'Weight_in_gms': [weight_in_gms],
            'Warehouse_block_A': [warehouse_block_A],
            'Warehouse_block_B': [warehouse_block_B],
            'Warehouse_block_C': [warehouse_block_C],
            'Warehouse_block_D': [warehouse_block_D],
            'Mode_of_Shipment_Flight': [mode_of_shipment_flight],
            'Mode_of_Shipment_Road': [mode_of_shipment_road]
        })

        # Use the loaded model to make predictions
        predictions = pipeline.predict(input_data)

        if predictions[0] == 1:
            return 'Product Reached With Delay'
        else:
            return 'Product Reached on Time'
    except Exception as e:
        # Handle any errors gracefully and return an error response
        return jsonify({'error': str(e)}), 400




if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Specify both host and port








