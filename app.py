from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load the trained model from train_data.pkl
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Encoding mappings for categorical variables
gender_mapping = {'Male': 1, 'Female': 0}
married_mapping = {'Yes': 1, 'No': 0}
dependents_mapping={'0' : 0,'1': 1,'2': 2,'3+': 3}
education_mapping = {'Graduate': 1, 'Not Graduate': 0}
property_area_mapping = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Map input values to match model's expected feature names
    input_data = {
        'Gender': request.form['Gender'],
        'Married': request.form['Married'],
        'Dependents': request.form['Dependents'],
        'Education' : request.form['Education'],
        'ApplicantIncome': float(request.form['ApplicantIncome']),
        'LoanAmount' : float(request.form['LoanAmount']),
        'Credit_History': float(request.form['Credit_History']),
        'Property_Area' :request.form['Property_Area']
    }
    
    # Convert input data to DataFrame with only the selected features
    input_df = pd.DataFrame([input_data])


    # Predict Loan Status
    prediction = model.predict(input_df)
    prediction_text = "Approved" if prediction[0] == 1 else "Rejected"
    
    # Display result on the page
    return render_template('index.html', prediction_text=f'Loan Prediction: {prediction_text}')


  

if __name__ == "__main__":
    app.run(debug=True)
