import pickle
import numpy as np
import shap

lifestyle_model = pickle.load(open('liver_lifestyle_model.pkl', 'rb'))
clinical_model = pickle.load(open('liver_clinical_model.pkl', 'rb'))
scaler = pickle.load(open('liver_scaler.pkl', 'rb'))

def test_shap():
    print("Testing Lifestyle...")
    features = np.array([[45, 1, 26.5, 10, 1, 1, 3.5, 0, 1, 60]])
    explainer_L = shap.TreeExplainer(lifestyle_model)
    shap_vals_L = explainer_L.shap_values(features)
    print("L type:", type(shap_vals_L))
    print("L shape:", np.array(shap_vals_L).shape)
    
    print("Testing Clinical...")
    blood_markers = [1.2, 0.4, 200, 35, 40, 7.5, 3.5, 0.9]
    scaled_markers = scaler.transform([blood_markers])[0]
    final_features = np.array([[45, 1] + list(scaled_markers)])
    explainer_C = shap.TreeExplainer(clinical_model)
    shap_vals_C = explainer_C.shap_values(final_features)
    print("C type:", type(shap_vals_C))
    print("C shape:", np.array(shap_vals_C).shape)

test_shap()
