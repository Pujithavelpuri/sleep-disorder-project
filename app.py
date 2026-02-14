# app.py - Complete Streamlit App with ML Integration

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Set page configuration
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="centered"
)

# Title and description
st.title("😴 Sleep Disorder Prediction System")
st.markdown(
    """
    This application predicts the type of sleep disorder (None, Insomnia, Sleep Apnea)  
    based on health and lifestyle indicators. Fill in the patient details below to get a prediction.
    """
)

# Function to load the trained model
@st.cache_resource
def load_model():
    """Load the trained Random Forest model and label encoders"""
    try:
        # Try to load pre-saved model if it exists
        model = joblib.load('best_sleep_model.pkl')
        feature_encoders = joblib.load('feature_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        return model, feature_encoders, target_encoder
    except:
        # If model doesn't exist, train a new one using the dataset
        st.warning("Training new model from dataset...")
        return train_new_model()

def train_new_model():
    """Train a new Random Forest model from the cleaned dataset"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv("sleep_cleaned.csv")
    
    # Prepare features and target
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    
    # Encode categorical variables
    feature_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        feature_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y_encoded)
    
    # Save model and encoders
    joblib.dump(model, 'best_sleep_model.pkl')
    joblib.dump(feature_encoders, 'feature_encoders.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    
    return model, feature_encoders, target_encoder

# Load the model
model, feature_encoders, target_encoder = load_model()

# Define expected features (from the training data)
expected_features = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 
    'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP'
]

# Create input form
st.header("📋 Enter Patient Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox(
        "Occupation",
        ["Software Engineer", "Doctor", "Sales Representative", "Teacher", 
         "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", "Manager"]
    )
    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, step=0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 6)
    physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 120, 60)

with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    
    # Blood Pressure input
    st.markdown("**Blood Pressure**")
    bp_col1, bp_col2 = st.columns(2)
    with bp_col1:
        systolic = st.number_input("Systolic", min_value=80, max_value=200, value=120)
    with bp_col2:
        diastolic = st.number_input("Diastolic", min_value=50, max_value=130, value=80)
    blood_pressure = f"{systolic}/{diastolic}"
    
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=5000, step=100)

# Prediction button
if st.button("🔍 Predict Sleep Disorder", type="primary"):
    
    # Prepare input data as DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic
    }])
    
    # Encode categorical variables using saved encoders
    try:
        # Encode Gender
        if 'Gender' in feature_encoders:
            input_data['Gender'] = feature_encoders['Gender'].transform(input_data['Gender'])
        
        # Encode Occupation
        if 'Occupation' in feature_encoders:
            input_data['Occupation'] = feature_encoders['Occupation'].transform(input_data['Occupation'])
        
        # Encode BMI Category
        if 'BMI Category' in feature_encoders:
            input_data['BMI Category'] = feature_encoders['BMI Category'].transform(input_data['BMI Category'])
        
        # Make prediction
        prediction_encoded = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Decode prediction
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Create result cards
        col1, col2, col3 = st.columns(3)
        
        # Define colors and icons for each prediction
        if prediction == "None":
            bg_color = "#28a745"
            icon = "✅"
            description = "No sleep disorder detected. Your sleep patterns appear normal."
        elif prediction == "Insomnia":
            bg_color = "#ffc107"
            icon = "🌙"
            description = "Insomnia detected. You may have difficulty falling or staying asleep."
        else:  # Sleep Apnea
            bg_color = "#dc3545"
            icon = "⚠️"
            description = "Sleep Apnea detected. Your breathing may be interrupted during sleep."
        
        # Display prediction with styling
        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin: 20px 0;
            ">
                <h1 style="font-size: 48px;">{icon}</h1>
                <h2>Prediction: {prediction}</h2>
                <p style="font-size: 16px;">{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display probability scores
        st.subheader("📊 Probability Scores")
        
        # Get class labels
        class_labels = target_encoder.classes_
        
        # Create probability DataFrame
        proba_df = pd.DataFrame({
            'Sleep Disorder': class_labels,
            'Probability': prediction_proba * 100
        })
        
        # Display as bar chart
        st.bar_chart(proba_df.set_index('Sleep Disorder'))
        
        # Display detailed probabilities
        for label, prob in zip(class_labels, prediction_proba):
            prob_percent = prob * 100
            if label == prediction:
                st.markdown(f"**{label}:** {prob_percent:.1f}% ⭐")
            else:
                st.markdown(f"**{label}:** {prob_percent:.1f}%")
        
        # Additional health recommendations
        st.subheader("💡 Health Recommendations")
        
        if prediction == "None":
            st.info("""
            - Maintain your current healthy sleep habits
            - Continue regular physical activity
            - Keep stress levels low with relaxation techniques
            """)
        elif prediction == "Insomnia":
            st.warning("""
            - Establish a consistent sleep schedule
            - Avoid caffeine and screen time before bed
            - Practice relaxation techniques like meditation
            - Consider consulting a sleep specialist
            """)
        else:  # Sleep Apnea
            st.error("""
            - Consult a healthcare provider immediately
            - Consider a sleep study for proper diagnosis
            - Maintain a healthy weight through diet and exercise
            - Sleep on your side instead of your back
            - Avoid alcohol before bedtime
            """)
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("Please ensure all fields are filled correctly.")

# Add information about the model
with st.expander("ℹ️ About the Model"):
    st.markdown("""
    **Model Information:**
    - **Algorithm:** Random Forest Classifier
    - **Features Used:** Age, Gender, Occupation, Sleep Duration, Quality of Sleep, 
      Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps
    - **Target Classes:** None, Insomnia, Sleep Apnea
    
    **Model Performance:**
    - The model was trained on health and lifestyle data
    - Achieved high accuracy in predicting sleep disorders
    - Uses ensemble learning for robust predictions
    
    **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

# Add sidebar with additional information
with st.sidebar:
    st.header("📌 Quick Tips for Better Sleep")
    st.markdown("""
    1. **Maintain a consistent sleep schedule**
    2. **Create a relaxing bedtime routine**
    3. **Avoid screens 1 hour before bed**
    4. **Limit caffeine after 2 PM**
    5. **Exercise regularly, but not too late**
    6. **Keep your bedroom cool and dark**
    7. **Manage stress with meditation**
    """)
    
    st.divider()
    
    st.header("📊 Dataset Statistics")
    try:
        df_stats = pd.read_csv("sleep_cleaned.csv")
        st.metric("Total Records", len(df_stats))
        st.metric("Features", len(df_stats.columns) - 1)
        
        # Show class distribution
        disorder_counts = df_stats['Sleep Disorder'].value_counts()
        st.bar_chart(disorder_counts)
    except:
        st.warning("Dataset not available for statistics")
