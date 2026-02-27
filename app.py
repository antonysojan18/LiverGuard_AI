from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime

import shap

app = Flask(__name__)

# Load the Super Ultra Assets we created
lifestyle_model = pickle.load(open('liver_lifestyle_model.pkl', 'rb'))
clinical_model = pickle.load(open('liver_clinical_model.pkl', 'rb'))
scaler = pickle.load(open('liver_scaler.pkl', 'rb'))

# Prepare SHAP explainers (do it once to save time)
lifestyle_explainer = shap.TreeExplainer(lifestyle_model)
clinical_explainer = shap.TreeExplainer(clinical_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/basic')
def basic():
    return render_template('basic.html')

@app.route('/advanced')
def advanced():
    return render_template('advanced.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        mode = data.get('mode')

        if mode == 'lifestyle':
            # Extract Lifestyle Inputs
            features = [
                int(data['age']), int(data['gender']), float(data['bmi']),
                float(data['alcohol']), int(data['smoking']), int(data['genetic']),
                float(data['activity']), int(data['diabetes']), int(data['hypertension']),
                float(data['lft']), float(data['sleep']), float(data['stress']),
                int(data['exposure']), float(data['upf']), float(data['hydration']),
                int(data['sugar'])
            ]
            feature_array = np.array([features])
            prediction = lifestyle_model.predict(feature_array)[0]
            probability = lifestyle_model.predict_proba(feature_array)[0][1]
            
            # Explainable AI - SHAP
            shap_vals = lifestyle_explainer.shap_values(feature_array)
            # RandomForest returns (samples, features, classes). We want class 1.
            feature_importance = shap_vals[0, :, 1].tolist()
            feature_names = [
                'Age', 'Gender', 'BMI', 'Alcohol', 'Smoking', 'Genetic Risk', 
                'Activity', 'Diabetes', 'Hypertension', 'LFT', 'Sleep Quality',
                'Stress Index', 'Enviro Exposure', 'Processed Food Intake', 'Hydration', 'Sugar Addiction'
            ]

        else:
            # Extract Clinical Inputs
            age = int(data['age_c'])
            gender = int(data['gender_c'])
            blood_markers = [
                float(data['tb']), float(data['db']), float(data['alp']),
                float(data['alt']), float(data['ast']), float(data['tp']),
                float(data['alb']), float(data['ag'])
            ]
            # Scale blood markers before prediction
            scaled_markers = scaler.transform([blood_markers])
            final_features = np.concatenate([[age, gender], scaled_markers[0]])
            feature_array = np.array([final_features])
            
            prediction = clinical_model.predict(feature_array)[0]
            probability = clinical_model.predict_proba(feature_array)[0][1]
            
            # Explainable AI - SHAP
            shap_vals = clinical_explainer.shap_values(feature_array)
            # GradientBoosting returns (samples, features)
            feature_importance = shap_vals[0, :].tolist()
            feature_names = ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 'ALP Level', 'ALT (SGPT)', 'AST (SGOT)', 'Total Proteins', 'Albumin', 'A/G Ratio']

        # Determine Low, Medium, High risk based on probability
        if probability < 0.40:
            risk_level = "Low"
            result = "Low Risk - Healthy"
        elif probability < 0.70:
            risk_level = "Medium"
            result = "Moderate Risk - Monitor Closely"
        else:
            risk_level = "High"
            result = "High Risk - Consult a Specialist"
            
        confidence = f"{round(probability * 100, 2)}%"
        
        # Generate Reasons and Diet Plan
        combined_shap = list(zip(feature_names, feature_importance))
        if risk_level == "High":
            combined_shap.sort(key=lambda x: x[1], reverse=True)
            reasons = [f"Critical Biomarker Detected: {item[0]} shows a significant deviation, substantially increasing the overall metabolic risk profile." for item in combined_shap[:3] if item[1] > 0]
            if not reasons:
                reasons = ["Systemic interaction of multiple borderline variables is compounding into a high-risk trajectory."]
                
            diet_plan = [
                "Implement a STRICT Mediterranean or hepatic-support diet (high in vegetables, omega-3s, and whole grains).",
                "Eliminate processed trans-fats and severely restrict all added simple sugars and high-fructose corn syrup.",
                "Prioritize high-quality lean proteins (e.g., wild-caught fish, poultry, legumes) to maintain muscle mass without overburdening the liver.",
                "Incorporate antioxidant-dense foods (dark berries, leafy greens, walnuts) to combat oxidative cellular stress."
            ]
            ai_recs = [
                "Complete abstinence from alcohol is strongly recommended to allow hepatic cellular regeneration.",
                "Maintain optimal hydration (2.5L - 3L daily) to assist renal and hepatic toxin clearance.",
                "Schedule a clinical follow-up immediately for advanced enzymatic and imaging diagnostics."
            ]
        elif risk_level == "Medium":
            combined_shap.sort(key=lambda x: x[1], reverse=True)
            reasons = [f"Elevated Factor: {item[0]} is moderately outside optimal bounds, contributing primarily to your elevated risk." for item in combined_shap[:3] if item[1] > 0]
            if not reasons:
                reasons = ["Marginal deviations across several metrics suggest early-stage metabolic stress."]
                
            diet_plan = [
                "Begin transitioning to a whole-foods-based diet, minimizing ultra-processed meals.",
                "Increase daily dietary fiber intake using complex carbohydrates (oats, brown rice, vegetables).",
                "Opt for white meat and plant-based proteins more often than red meat.",
                "Avoid late-night meals to reduce metabolic stress before sleep."
            ]
            ai_recs = [
                "Limit alcohol consumption strictly to occasional, minimal quantities, or abstain completely.",
                "Establish a routine of at least 150 minutes of moderate-intensity cardiovascular exercise per week.",
                "Increase daily hydration to adequately support metabolic processes.",
                "Optimize circadian rhythms by prioritizing 7-8 hours of uninterrupted, high-quality sleep.",
                "Monitor and actively engage in stress-reduction techniques to lower systemic cortisol."
            ]
        else:
            combined_shap.sort(key=lambda x: x[1])
            reasons = [f"Protective Factor: Your optimal levels in {item[0]} strongly correlate with a healthy metabolic state." for item in combined_shap[:3] if item[1] < 0]
            if not reasons:
                reasons = ["All major metabolic and lifestyle indicators are well within optimal clinical thresholds."]
            
            diet_plan = [
                "Maintain your current balanced diet—your nutritional habits are effectively supporting hepatic function.",
                "Continue to prioritize natural, whole foods while successfully avoiding excessive simple sugars.",
                "Continue maintaining a healthy balance of macro-nutrients, particularly healthy fats and lean proteins."
            ]
            ai_recs = [
                "Sustain your current level of physical activity to ensure ongoing cardiovascular and metabolic health.",
                "Keep alcohol consumption strictly moderate or continue with total abstinence.",
                "Ensure your daily hydration remains consistent.",
                "Schedule standard annual check-ups to maintain this excellent health profile."
            ]
        
        # Save record to CSV
        record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mode': mode,
            'prediction': risk_level,
            'confidence': confidence
        }
        
        all_columns = [
            'timestamp', 'mode', 'prediction', 'confidence',
            'patient_name', 'age', 'gender', 'bmi', 'alcohol', 'smoking', 'genetic', 'activity', 'diabetes', 'hypertension', 'lft',
            'sleep', 'stress', 'exposure', 'upf', 'hydration', 'sugar',
            'age_c', 'gender_c', 'tb', 'db', 'alp', 'alt', 'ast', 'tp', 'alb', 'ag'
        ]
        
        row_data = {}
        form_dict = data.to_dict()
        for col in all_columns:
            if col in record:
                row_data[col] = record[col]
            else:
                row_data[col] = form_dict.get(col, "N/A")
                
        csv_file = 'patient_record.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        # Render different templates based on mode
        template_name = 'basic_result.html' if mode == 'lifestyle' else 'advanced_result.html'

        return render_template(template_name, 
                               prediction_text=result, 
                               conf=confidence, 
                               risk_level=risk_level,
                               mode=mode, 
                               data=data.to_dict(),
                               feature_names=feature_names,
                               feature_importance=feature_importance,
                               reasons=reasons,
                               diet_plan=diet_plan,
                               ai_recs=ai_recs)

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/consult')
def consult():
    try:
        df_docs = pd.read_csv('south_india_liver_doctors.csv')
        doctors = df_docs.to_dict('records')
        return render_template('doctors.html', doctors=doctors)
    except Exception as e:
        return str(e)

@app.route('/book/<doc_id>')
def book(doc_id):
    try:
        df_docs = pd.read_csv('south_india_liver_doctors.csv')
        doctor = df_docs[df_docs['Doctor_ID'] == doc_id].to_dict('records')
        if not doctor:
            return "Doctor not found"
        return render_template('booking.html', doctor=doctor[0], now=datetime.now())
    except Exception as e:
        return str(e)

@app.route('/payment', methods=['POST'])
def payment():
    details = request.form.to_dict()
    return render_template('payment.html', details=details)

@app.route('/confirmation', methods=['POST'])
def confirmation():
    details = request.form.to_dict()
    import uuid
    tx_id = str(uuid.uuid4()).split('-')[0].upper()
    
    # Save booking to CSV
    csv_file = 'hospital_records.csv'
    file_exists = os.path.isfile(csv_file)
    
    record = details.copy()
    record['Transaction_ID'] = tx_id
    record['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fieldnames = [
        'Transaction_ID', 'Timestamp', 'patient_name', 'phone', 
        'doc_name', 'hospital', 'date', 'time', 
        'method', 'payment_amount', 'upi_app'
    ]
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
        
    return render_template('confirmation.html', details=details, tx=tx_id)

if __name__ == "__main__":
    app.run(debug=True)