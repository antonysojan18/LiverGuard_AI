import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# 1. DATA PREPROCESSING MODULE (ETL & CLEANING)
# ==========================================================
def load_and_preprocess():
    print("\n[1/6] Initializing Data Preprocessing...")
    
    # Load the Augmented & Balanced 2000-record datasets
    try:
        df_clinical = pd.read_csv('ilpd_balanced_2000.csv')
        df_lifestyle = pd.read_csv('lifestyle_balanced_2000.csv')
    except FileNotFoundError:
        print("ERROR: Balanced CSV files not found. Ensure filenames match exactly.")
        return None

    # --- THE CRITICAL FIX: HANDLING NaN AND ALIGNING TARGETS ---
    # Removes any empty rows that caused your previous error
    df_clinical.dropna(inplace=True)
    df_lifestyle.dropna(inplace=True)

    # Standardize Clinical Target (Ensure 1=Disease, 0=Healthy)
    if 'Dataset' in df_clinical.columns:
        # If the file still has 2 for healthy, map it. If it's already 0, this stays safe.
        df_clinical['Dataset'] = df_clinical['Dataset'].replace(2, 0)
    
    # Standardize Lifestyle Target Name
    if 'Diagnosis' in df_lifestyle.columns:
        df_lifestyle.rename(columns={'Diagnosis': 'Result'}, inplace=True)
    elif 'Result' not in df_lifestyle.columns:
        # Fallback if the column name is different in your augmented file
        df_lifestyle.rename(columns={df_lifestyle.columns[-1]: 'Result'}, inplace=True)

    # --- Feature Engineering ---
    # Encode Gender in Clinical data
    le = LabelEncoder()
    if df_clinical['Gender'].dtype == 'object':
        df_clinical['Gender'] = le.fit_transform(df_clinical['Gender'])

    # Scaling Clinical Markers (Crucial for M.Sc. level accuracy)
    scaler = StandardScaler()
    clinical_markers = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
                        'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
    
    df_clinical[clinical_markers] = scaler.fit_transform(df_clinical[clinical_markers])

    print(f"Data ready. Clinical samples: {len(df_clinical)}, Lifestyle samples: {len(df_lifestyle)}")
    return df_clinical, df_lifestyle, scaler

# ==========================================================
# 2. HYPERPARAMETER OPTIMIZATION (GRID SEARCH)
# ==========================================================
def optimize_engines(X_l, y_l, X_c, y_c):
    print("\n[2/6] Optimizing Hyperparameters for Peak Performance...")

    # Tuning Engine A: Lifestyle (Random Forest)
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'criterion': ['gini', 'entropy']
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3)
    grid_rf.fit(X_l, y_l)
    
    # Tuning Engine B: Clinical (Gradient Boosting)
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3)
    grid_gb.fit(X_c, y_c)

    print("Optimization Complete.")
    return grid_rf.best_estimator_, grid_gb.best_estimator_

# ==========================================================
# 3. SCIENTIFIC VALIDATION MODULE
# ==========================================================
def validate_system(m_l, m_c, X_t_l, y_t_l, X_t_c, y_t_c):
    print("\n[3/6] Generating Scientific Validation Metrics...")
    
    for name, model, X_test, y_test in [("Lifestyle", m_l, X_t_l, y_t_l), ("Clinical", m_c, X_t_c, y_t_c)]:
        y_pred = model.predict(X_test)
        print(f"\n--- {name} Engine Classification Report ---")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title(f'Confusion Matrix: {name} Engine')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_{name}.png')
        plt.close()

# ==========================================================
# 4. EXPORT MODULE (MODEL PERSISTENCE)
# ==========================================================
def export_assets(m_l, m_c, scaler):
    print("\n[4/6] Exporting Serialized Assets (.pkl)...")
    pickle.dump(m_l, open('liver_lifestyle_model.pkl', 'wb'))
    pickle.dump(m_c, open('liver_clinical_model.pkl', 'wb'))
    pickle.dump(scaler, open('liver_scaler.pkl', 'wb'))
    print("Assets saved successfully.")

# ==========================================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================================
if __name__ == "__main__":
    # A. Preprocess
    data = load_and_preprocess()
    if data:
        df_c, df_l, main_scaler = data

        # B. Split Data
        X_l = df_l.drop('Result', axis=1)
        y_l = df_l['Result']
        X_c = df_c.drop('Dataset', axis=1)
        y_c = df_c['Dataset']

        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2, random_state=42)
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

        # C. Optimize & Train
        best_lifestyle, best_clinical = optimize_engines(X_train_l, y_train_l, X_train_c, y_train_c)

        # D. Validate
        validate_system(best_lifestyle, best_clinical, X_test_l, y_test_l, X_test_c, y_test_c)

        # E. Export
        export_assets(best_lifestyle, best_clinical, main_scaler)

        print("\n[6/6] PROJECT DEPLOYMENT READY.")