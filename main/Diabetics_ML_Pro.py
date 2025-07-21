import pandas as pd
try:
    data = pd.read_csv('/content/diabetes.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: diabetes.csv not found. Make sure you uploaded the file.")
except Exception as e:
    print(f"An error occurred during dataset loading: {e}")
print(data.head())
import numpy as np
try:
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[zero_cols] = data[zero_cols].replace(0, np.nan)
    for col in zero_cols:
        data[col] = data[col].fillna(data[col].median())
    print("Preprocessing completed successfully!")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
# Feature Engineering for Future Risk Prediction
data['AgeSquared'] = data['Age'] ** 2
data['GlucoseAgeInteraction'] = data['Glucose'] * data['Age']
data['BMIChange'] = data['BMI'] - 25
data['YearsToPredict'] = 5
data['AgePlusYears'] = data['Age'] + data['YearsToPredict']
# Future Outcome Simulation
import numpy as np
non_diabetics = data[data['Outcome'] == 0].copy()
np.random.seed(42)
non_diabetics['FutureOutcome'] = np.random.choice([0, 1], size=len(non_diabetics), p=[0.7, 0.3])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(data.columns)
from sklearn.model_selection import train_test_split
X_future = non_diabetics.drop(['Outcome', 'FutureOutcome'], axis=1)
y_future = non_diabetics['FutureOutcome']
X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(X_future, y_future, test_size=0.2, random_state=42)
print(f"X_train_future shape: {X_train_future.shape}, X_test_future shape: {X_test_future.shape}")
print(non_diabetics.columns)
#Feature Scaling for Future Risk Model
from sklearn.preprocessing import StandardScaler
scaler_future = StandardScaler()
X_train_future_scaled = scaler_future.fit_transform(X_train_future)
X_test_future_scaled = scaler_future.transform(X_test_future)
#6. Train Standard Diabetes Prediction Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

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
        'SVM': SVC(random_state=42)
    }

    model_accuracies = {}
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
        print(f'{name} Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"Best model: {best_model.__class__.__name__} with accuracy: {best_accuracy:.4f}")
    diabetes_model = best_model
    with open("diabetes_model.pkl", "wb") as f:
        pickle.dump(diabetes_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

except Exception as e:
    print(f"Error training and saving diabetes model: {e}")
# Train and Compare Future Risk Prediction Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

model_accuracies = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_future_scaled, y_train_future)
    y_pred_future = model.predict(X_test_future_scaled)
    accuracy = accuracy_score(y_test_future, y_pred_future)
    model_accuracies[name] = accuracy
    print(f'{name} Accuracy: {accuracy:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"Best model: {best_model.__class__.__name__} with accuracy: {best_accuracy:.4f}")
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
# UI using ipywidgets
glucose_input = widgets.FloatText(description="Glucose")
blood_pressure_input = widgets.FloatText(description="Blood Pressure")
skin_thickness_input = widgets.FloatText(description="Skin Thickness")
insulin_input = widgets.FloatText(description="Insulin")
bmi_input = widgets.FloatText(description="BMI")
diabetes_pedigree_input = widgets.FloatText(description="Diabetes Pedigree")
age_input = widgets.IntText(description="Age")

predict_button = widgets.Button(description="Predict Diabetes")
future_risk_button = widgets.Button(description="Predict Future Risk")
output_area = widgets.Output()

def predict_diabetes(button):
    with output_area:
        clear_output()
        print("Diabetes Prediction:")
        try:
            age = age_input.value
            years_to_predict = 5
            age_squared = age ** 2
            glucose_age_interaction = glucose_input.value * age
            bmi_change = bmi_input.value - 25
            age_plus_years = age + years_to_predict

            input_data = pd.DataFrame(
                [[glucose_input.value, blood_pressure_input.value, skin_thickness_input.value, insulin_input.value,
                  bmi_input.value, diabetes_pedigree_input.value, age, age_squared, glucose_age_interaction,
                  bmi_change, years_to_predict, age_plus_years]],
                columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                         'Age', 'AgeSquared', 'GlucoseAgeInteraction', 'BMIChange',
                         'YearsToPredict', 'AgePlusYears']
            )

            input_data_scaled = scaler.transform(input_data)
            prediction = diabetes_model.predict(input_data_scaled)[0]
            if prediction == 1:
                print("The person is Diabetic")
                future_risk_button.disabled = True
            else:
                print("The person is Diabetic Free")
                future_risk_button.disabled = False
        except Exception as e:
            print(f"Error: {e}")

def predict_future_risk(button):
    with output_area:
        clear_output()
        print("Future Risk Prediction:")
        try:
            age = age_input.value
            years_to_predict = 5
            age_squared = age ** 2
            glucose_age_interaction = glucose_input.value * age
            bmi_change = bmi_input.value - 25
            age_plus_years = age + years_to_predict

            input_data = pd.DataFrame(
                [[glucose_input.value, blood_pressure_input.value, skin_thickness_input.value, insulin_input.value,
                  bmi_input.value, diabetes_pedigree_input.value, age, age_squared, glucose_age_interaction,
                  bmi_change, years_to_predict, age_plus_years]],
                columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                         'Age', 'AgeSquared', 'GlucoseAgeInteraction', 'BMIChange',
                         'YearsToPredict', 'AgePlusYears']
            )

            input_scaled = scaler_future.transform(input_data)
            risk_score = best_model.predict_proba(input_scaled)[0][1]
            print(f"Risk of diabetes in 5 years: {risk_score * 100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")

predict_button.on_click(predict_diabetes)
future_risk_button.on_click(predict_future_risk)

display(glucose_input, blood_pressure_input, skin_thickness_input, insulin_input,
        bmi_input, diabetes_pedigree_input, age_input,
        predict_button, future_risk_button, output_area)
