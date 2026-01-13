from flask import Flask, request, render_template, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

app.secret_key = 'super_secret_key_for_churn_app'

# Load trained files
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    prediction_text = session.pop('prediction_text', None)
    result_color = session.pop('result_color', None)  
    form_data = session.pop('form_data', None)
    
    return render_template('index.html', 
                           prediction_text=prediction_text, 
                           result_color=result_color, 
                           form_data=form_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Tenure': int(request.form['Tenure']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
            'Dependents': request.form['Dependents'],
            'PhoneService': request.form['PhoneService'],
            'InternetService': request.form['InternetService'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod']
        }
        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input)
        df_final = df_input.reindex(columns=model_columns, fill_value=0)
        df_final = scaler.transform(df_final)
        
        prediction = model.predict(df_final)
        probability = model.predict_proba(df_final)[0][1] * 100
        
        if prediction[0] == 1:
            result = "Likely to CHURN"
            result_color = "churn-danger" 
        else:
            result = "Likely to STAY"
            result_color = "safe-success" 
        
        session['prediction_text'] = f'{result} ({probability:.2f}% Risk)'
        session['result_color'] = result_color  
        session['form_data'] = input_data 
        
        return redirect(url_for('home'))
        
    except Exception as e:
        session['prediction_text'] = f'Error: {str(e)}'
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')