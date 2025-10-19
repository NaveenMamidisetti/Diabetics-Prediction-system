import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
try:
    data = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: diabetes.csv not found. Make sure you uploaded the file.")
except Exception as e:
    print(f"An error occurred during dataset loading: {e}")

print(data.head())

# 2. Data Preprocessing
try:
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[zero_cols] = data[zero_cols].replace(0, np.nan)
    for col in zero_cols:
        data[col] = data[col].fillna(data[col].median())
    print("Preprocessing completed successfully!")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")

# 3. Feature Engineering for Future Risk Prediction
# IMPORTANT: We need to include ALL original features plus engineered ones
data['AgeSquared'] = data['Age'] ** 2
data['GlucoseAgeInteraction'] = data['Glucose'] * data['Age']
data['BMIChange'] = data['BMI'] - 25
data['YearsToPredict'] = 5
data['AgePlusYears'] = data['Age'] + data['YearsToPredict']

print("✅ Feature engineering completed!")
print(f"Total features after engineering: {len(data.columns)}")
print(f"Features: {list(data.columns)}")

# 4. REALISTIC Future Outcome Simulation (Improved Method)
def create_realistic_future_labels(data):
    """
    Create realistic future diabetes outcomes based on medical risk factors
    Instead of random 30% for everyone, use personalized risks
    """
    labels = []
    
    for _, patient in data.iterrows():
        base_risk = 0
        
        # Medical risk factors (based on real clinical knowledge)
        if patient['Glucose'] > 140: base_risk += 0.3      # High glucose = major risk
        if patient['BMI'] > 30: base_risk += 0.25          # Obesity = significant risk
        if patient['Age'] > 50: base_risk += 0.15          # Age > 50 = increased risk
        if patient['BloodPressure'] > 140: base_risk += 0.1 # High BP = additional risk
        if patient['DiabetesPedigreeFunction'] > 0.6: base_risk += 0.1 # Family history
        
        # Add some biological variation (real life has uncertainty)
        final_risk = base_risk + np.random.uniform(-0.08, 0.08)
        
        # Ensure reasonable bounds (2% to 60% risk)
        final_risk = max(0.02, min(final_risk, 0.6))
        
        # Generate outcome based on calculated risk probability
        labels.append(1 if np.random.random() < final_risk else 0)
    
    return labels

# Get non-diabetic patients for future risk prediction
non_diabetics = data[data['Outcome'] == 0].copy()
print(f"👥 Non-diabetic patients available: {len(non_diabetics)}")

# Create realistic future outcomes
print("🎯 Generating realistic future diabetes outcomes...")
np.random.seed(42)  # For reproducibility
non_diabetics['FutureOutcome'] = create_realistic_future_labels(non_diabetics)

# Check the distribution
future_diabetes_rate = non_diabetics['FutureOutcome'].mean()
print(f"📈 Realistic future diabetes rate: {future_diabetes_rate:.1%} (was 30% with random)")
print("✅ Realistic training labels generated!")

# 5. Prepare Data for Current Diabetes Prediction
# Make sure we include ALL features including the original ones
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Current diabetes model features: {X_train.shape[1]} features")

# 6. Prepare Data for Future Risk Prediction
X_future = non_diabetics.drop(['Outcome', 'FutureOutcome'], axis=1)
y_future = non_diabetics['FutureOutcome']
X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(
    X_future, y_future, test_size=0.2, random_state=42, stratify=y_future
)

print(f"X_train_future shape: {X_train_future.shape}, X_test_future shape: {X_test_future.shape}")
print(f"Future diabetes rate in training: {y_train_future.mean():.1%}")
print(f"Future diabetes rate in test: {y_test_future.mean():.1%}")

# Feature Scaling for Future Risk Model
scaler_future = StandardScaler()
X_train_future_scaled = scaler_future.fit_transform(X_train_future)
X_test_future_scaled = scaler_future.transform(X_test_future)

print(f"✅ Future risk model features: {X_train_future.shape[1]} features")

# 7. Train Standard Diabetes Prediction Model
try:
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }

    model_accuracies = {}
    best_model = None
    best_accuracy = 0

    print("🤖 Training Current Diabetes Prediction Models:")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
        print(f'{name} Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"🏆 Best model: {best_model.__class__.__name__} with accuracy: {best_accuracy:.4f}")
    diabetes_model = best_model
    
    # Save models
    with open("diabetes_model.pkl", "wb") as f:
        pickle.dump(diabetes_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("💾 Current diabetes model saved successfully!")

except Exception as e:
    print(f"Error training and saving diabetes model: {e}")

# 8. Train and Compare Future Risk Prediction Models (with realistic data)
print("\n🤖 Training Future Risk Prediction Models (Realistic Data):")

future_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

future_model_accuracies = {}
best_future_model = None
best_future_accuracy = 0

for name, model in future_models.items():
    model.fit(X_train_future_scaled, y_train_future)
    y_pred_future = model.predict(X_test_future_scaled)
    accuracy = accuracy_score(y_test_future, y_pred_future)
    future_model_accuracies[name] = accuracy
    print(f'{name} Accuracy: {accuracy:.4f}')

    if accuracy > best_future_accuracy:
        best_future_accuracy = accuracy
        best_future_model = model

print(f"🏆 Best future risk model: {best_future_model.__class__.__name__} with accuracy: {best_future_accuracy:.4f}")

# Save future risk model
with open("future_risk_model.pkl", "wb") as f:
    pickle.dump(best_future_model, f)
with open("scaler_future.pkl", "wb") as f:
    pickle.dump(scaler_future, f)
print("💾 Future risk model saved successfully!")

# 9. Feature Importance Analysis for Future Risk Model
if hasattr(best_future_model, 'feature_importances_'):
    print("\n🔬 Future Risk Model Feature Importance:")
    importances = best_future_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_future.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    print(feature_importance_df)

# 10. Simple Rule-Based Predictor (for comparison)
def simple_risk_predictor(patient_data):
    """
    Simple transparent risk predictor based on medical rules
    """
    glucose = patient_data['Glucose']
    bmi = patient_data['BMI']
    age = patient_data['Age']
    bp = patient_data['BloodPressure']
    family_history = patient_data['DiabetesPedigreeFunction']
    
    base_risk = 0
    
    # Same medical rules used for training
    if glucose > 140: base_risk += 0.3
    if bmi > 30: base_risk += 0.25
    if age > 50: base_risk += 0.15
    if bp > 140: base_risk += 0.1
    if family_history > 0.6: base_risk += 0.1
    
    # Cap the risk
    final_risk = min(base_risk, 0.6)
    
    return final_risk

# 11. UI using ipywidgets - FIXED VERSION
print("\n🎮 Launching Interactive Prediction Interface...")

# Define the CORRECT feature order based on our training data
CORRECT_FEATURE_ORDER = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'AgeSquared', 
    'GlucoseAgeInteraction', 'BMIChange', 'YearsToPredict', 'AgePlusYears'
]

glucose_input = widgets.FloatText(description="Glucose", value=120)
blood_pressure_input = widgets.FloatText(description="Blood Pressure", value=80)
skin_thickness_input = widgets.FloatText(description="Skin Thickness", value=20)
insulin_input = widgets.FloatText(description="Insulin", value=80)
bmi_input = widgets.FloatText(description="BMI", value=25)
diabetes_pedigree_input = widgets.FloatText(description="Diabetes Pedigree", value=0.5)
age_input = widgets.IntText(description="Age", value=35)
pregnancies_input = widgets.IntText(description="Pregnancies", value=1)  # Added missing feature

predict_button = widgets.Button(description="Predict Current Diabetes", button_style='primary')
future_risk_button = widgets.Button(description="Predict Future Risk (5 years)", button_style='info')
output_area = widgets.Output()

def create_input_dataframe():
    """Create input dataframe with ALL features in CORRECT order"""
    age = age_input.value
    years_to_predict = 5
    
    # Calculate engineered features
    age_squared = age ** 2
    glucose_age_interaction = glucose_input.value * age
    bmi_change = bmi_input.value - 25
    age_plus_years = age + years_to_predict
    
    # Create data in EXACT same order as training
    input_dict = {
        'Pregnancies': pregnancies_input.value,
        'Glucose': glucose_input.value,
        'BloodPressure': blood_pressure_input.value,
        'SkinThickness': skin_thickness_input.value,
        'Insulin': insulin_input.value,
        'BMI': bmi_input.value,
        'DiabetesPedigreeFunction': diabetes_pedigree_input.value,
        'Age': age,
        'AgeSquared': age_squared,
        'GlucoseAgeInteraction': glucose_age_interaction,
        'BMIChange': bmi_change,
        'YearsToPredict': years_to_predict,
        'AgePlusYears': age_plus_years
    }
    
    # Ensure correct order
    input_data = pd.DataFrame([input_dict])[CORRECT_FEATURE_ORDER]
    return input_data

def predict_diabetes(button):
    with output_area:
        clear_output()
        print("🔍 CURRENT DIABETES PREDICTION:")
        print("=" * 40)
        try:
            # Create input data with ALL features in correct order
            input_data = create_input_dataframe()
            
            print(f"📊 Input features: {len(input_data.columns)}")
            print(f"🔧 Features used: {list(input_data.columns)}")
            
            input_data_scaled = scaler.transform(input_data)
            prediction = diabetes_model.predict(input_data_scaled)[0]
            probability = diabetes_model.predict_proba(input_data_scaled)[0][1]
            
            if prediction == 1:
                print("❌ Result: The person is LIKELY DIABETIC")
                print(f"📊 Confidence: {probability:.1%}")
                future_risk_button.disabled = True
                print("💡 Future risk prediction disabled (person is already diabetic)")
            else:
                print("✅ Result: The person is DIABETIC FREE")
                print(f"📊 Confidence: {(1-probability):.1%}")
                future_risk_button.disabled = False
                print("💡 Future risk prediction enabled")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure all input fields are filled correctly")

def predict_future_risk(button):
    with output_area:
        clear_output()
        print("🔮 FUTURE RISK PREDICTION (5 years):")
        print("=" * 45)
        try:
            # Create input data with ALL features in correct order
            input_data = create_input_dataframe()
            
            print(f"📊 Input features: {len(input_data.columns)}")
            print(f"🔧 Features used: {list(input_data.columns)}")
            
            input_scaled = scaler_future.transform(input_data)
            risk_score = best_future_model.predict_proba(input_scaled)[0][1]
            
            # Also get simple rule-based prediction
            simple_risk = simple_risk_predictor(input_data.iloc[0])
            
            print(f"🤖 ML Model Prediction: {risk_score * 100:.1f}% risk")
            print(f"📋 Rule-Based Estimate: {simple_risk * 100:.1f}% risk")
            print(f"👤 Current Age: {age_input.value} → Future Age: {age_input.value + 5}")
            
            # Risk interpretation
            if risk_score < 0.1:
                print("💚 Risk Level: LOW - Maintain healthy lifestyle!")
            elif risk_score < 0.3:
                print("💛 Risk Level: MODERATE - Monitor regularly")
            else:
                print("❤️ Risk Level: HIGH - Consult healthcare provider")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure all input fields are filled correctly")

predict_button.on_click(predict_diabetes)
future_risk_button.on_click(predict_future_risk)

# Display the UI
print("\n" + "="*60)
print("🎯 DIABETES PREDICTION SYSTEM READY!")
print("="*60)
print("💡 Enter patient data and click buttons for predictions")
print("   - 'Predict Current Diabetes': Checks current diabetes status")
print("   - 'Predict Future Risk': Estimates 5-year diabetes risk")
print("")

display(pregnancies_input, glucose_input, blood_pressure_input, skin_thickness_input, insulin_input,
        bmi_input, diabetes_pedigree_input, age_input,
        predict_button, future_risk_button, output_area)

print("\n" + "="*60)
print("✅ SYSTEM SUMMARY:")
print(f"   • Current diabetes model: {diabetes_model.__class__.__name__}")
print(f"   • Future risk model: {best_future_model.__class__.__name__}")
print(f"   • Realistic future diabetes rate: {future_diabetes_rate:.1%}")
print(f"   • Future risk model accuracy: {best_future_accuracy:.3f}")
print(f"   • Total features used: {len(CORRECT_FEATURE_ORDER)}")
print("="*60)
