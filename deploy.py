import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
# ---------------------------------------
# Load the dataset
try:
    data = pd.read_csv('dataset/yield_df.csv')  # Update the path if needed
except FileNotFoundError:
    st.error("Dataset not found. Please ensure the file 'yield_df.csv' is in the 'dataset/' folder.")
    st.stop()

# Strip whitespace from string columns
data = data.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

# Display the dataset
st.subheader("Dataset Overview")
st.dataframe(data)

# Convert categorical features to numerical using Label Encoding
categorical_columns = ['Area', 'Item']  # Adjust these names as per your dataset
label_encoders = {}

for column in categorical_columns:
    if column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    else:
        st.warning(f"Column '{column}' not found in the dataset. Skipping.")

# Display the processed dataset
st.subheader("Processed Dataset")
st.dataframe(data)

# ---------------------------------------
# Load saved preprocessing tools
scaler = joblib.load('model1/scaler.pkl')
feature_names = joblib.load('model1/feature_names.pkl')
label_encoders = {
    'Item': joblib.load('model1/label_encoder_Item.pkl'),
    'Area': joblib.load('model1/label_encoder_Area.pkl')
}

# Streamlit user inputs
st.sidebar.header("User Input Parameters")
area = st.sidebar.selectbox("Select Area", label_encoders['Area'].classes_)
crop = st.sidebar.selectbox("Select Crop", label_encoders['Item'].classes_)
rainfall = st.sidebar.number_input("Average Rainfall (mm per year)", min_value=0.0, value=1500.0)
pesticides = st.sidebar.number_input("Pesticides (tonnes)", min_value=0.0, value=100.0)
temperature = st.sidebar.number_input("Average Temperature (°C)", min_value=-10.0, value=25.0)

# Allow user to select a model
models_available = {
    'Gradient Boost': 'gradient_boost_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'Linear Regression': 'linear_regression_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'KNN': 'knn_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'Bagging Regressor': 'bagging_regressor_model.pkl',
    'Average of All Models': 'average_of_all_models'  # New option for average model
}

model_name = st.sidebar.selectbox("Select Model", list(models_available.keys()))

# Prepare input data
input_data = pd.DataFrame({
    'average_rain_fall_mm_per_year': [rainfall],
    'pesticides_tonnes': [pesticides],
    'avg_temp': [temperature],
    'Item': [label_encoders['Item'].transform([crop])[0]],
    'Area': [label_encoders['Area'].transform([area])[0]]
})

# Ensure columns match the expected feature names
try:
    input_data = input_data[feature_names]  # Align columns to feature names
    scaled_input = scaler.transform(input_data)
except KeyError as e:
    st.error(f"Error: Missing or misaligned columns in input data: {e}")
    st.stop()
except ValueError as e:
    st.error(f"Error in scaling input data: {e}")
    st.stop()

# Function to load all models and make predictions
def load_and_predict(model_names, scaled_input):
    predictions = []
    for model_name in model_names:
        model_filepath = f'model1/{models_available[model_name]}'
        try:
            model = joblib.load(model_filepath)
            pred = model.predict(scaled_input)[0]
            predictions.append(pred)
        except FileNotFoundError:
            st.error(f"Error: Model file '{model_filepath}' not found.")
            st.stop()
    return predictions

# If the "Average of All Models" is selected
if model_name == 'Average of All Models':
    # List of all models excluding the 'Average of All Models' option
    all_model_names = [key for key in models_available.keys() if key != 'Average of All Models']
    
    # Get predictions from all models
    all_predictions = load_and_predict(all_model_names, scaled_input)
    
    # Calculate average prediction
    average_prediction = sum(all_predictions) / len(all_predictions)
    
    # Display average prediction
    st.subheader("Predicted Crop Yield (Average of All Models)")
    st.write(f"The predicted crop yield using **Average of All Models** is: **{average_prediction:.2f} hg/ha_yield**")
else:
    # Load the selected model
    model_filepath = f'model1/{models_available[model_name]}'
    try:
        model = joblib.load(model_filepath)
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filepath}' not found.")
        st.stop()

    # Make prediction
    predicted_yield = model.predict(scaled_input)[0]

    # Display prediction
    st.subheader("Predicted Crop Yield")
    st.write(f"The predicted crop yield using **{model_name}** is: **{predicted_yield:.2f} hg/ha_yield**")
