from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('D:\\supply chain\\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting all 31 features from the form
        input_features = [
            float(request.form['Product type']),
            float(request.form['Price']),
            float(request.form['Availability']),
            float(request.form['Number of products sold']),
            float(request.form['Revenue generated']),
            float(request.form['Customer demographics']),
            float(request.form['Lead times']),
            float(request.form['Order quantities']),
            float(request.form['Shipping times']),
            float(request.form['Shipping carriers']),
            float(request.form['Shipping costs']),
            float(request.form['Supplier name']),
            float(request.form['Location']),
            float(request.form['Lead time']),
            float(request.form['Production volumes']),
            float(request.form['Manufacturing lead time']),
            float(request.form['Manufacturing costs']),
            float(request.form['Defect rates']),
            float(request.form['Transportation modes']),
            float(request.form['Routes']),
            float(request.form['Costs']),
            float(request.form['Stock_Level_t-1']),
            float(request.form['Stock_Level_t-2']),
            float(request.form['Stock_Level_MA_3']),
            float(request.form['Stock_Level_MA_7']),
            float(request.form['Lead_Time_Demand']),
            float(request.form['Reorder_Point']),
            float(request.form['Avg_Stock_Level']),
            float(request.form['Stock_Turnover_Ratio']),
            float(request.form['COGS_per_day']),
            float(request.form['DIO']),
        ]

        # Reshape input to match model's expected shape
        input_array = np.array(input_features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(input_array)[0]

        return render_template('index.html', prediction_text=f'Predicted Stock Level: {prediction}')

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
