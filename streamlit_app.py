import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import WaterCapacityModel

# Set page config
st.set_page_config(
    page_title="Water Capacity Prediction", 
    layout="wide",
    page_icon="💧"
)

# Sidebar navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", [
    "Prediksi", 
    "Visualisasi Data", 
    "Evaluasi Model"
])

# Load the model
@st.cache_resource
def load_model():
    model = WaterCapacityModel()
    model.load_state_dict(torch.load('water_capacity_model.pth'))
    model.eval()
    return model

# Load the scaler
@st.cache_resource
def load_scaler():
    data = pd.read_csv('irrigation_water_prediction_dataset_300 (1).csv')
    X = data[['Kelembapan Tanah (%)', 'Luas Lahan (ha)', 'Curah Hujan (mm)']].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

# Data visualization page
def visualization_page():
    st.title("Visualisasi Data")
    data = pd.read_csv('irrigation_water_prediction_dataset_300 (1).csv')
    
    # Correlation heatmap
    st.subheader("Korelasi Antar Variabel")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.pyplot(fig)
    
    # Scatter plot matrix
    st.subheader("Matriks Scatter Plot")
    fig2 = plt.figure(figsize=(12, 8))
    pd.plotting.scatter_matrix(data, figsize=(12, 8), diagonal='kde')
    st.pyplot(fig2)
    
    # Feature distribution
    st.subheader("Distribusi Fitur")
    selected_feature = st.selectbox("Pilih parameter untuk visualisasi distribusi", 
                                  ['Kelembapan Tanah (%)', 'Luas Lahan (ha)', 'Curah Hujan (mm)', 'Kapasitas Air (m3)'])
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(data[selected_feature], kde=True, ax=ax3)
    ax3.set_title(f'Distribusi {selected_feature}')
    st.pyplot(fig3)

# Model evaluation page
def evaluation_page():
    st.title("Evaluasi Model")
    st.write("Evaluasi performa model prediksi kapasitas air")
    
    # Load test data
    data = pd.read_csv('irrigation_water_prediction_dataset_300 (1).csv')
    X = data[['Kelembapan Tanah (%)', 'Luas Lahan (ha)', 'Curah Hujan (mm)']].values
    y = data['Kapasitas Air (m3)'].values
    
    # Load model and scaler
    model = load_model()
    scaler = load_scaler()
    
    # Make predictions
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - y))
    mse = np.mean((predictions - y)**2)
    r2 = r2_score(y, predictions)
    
    # Display metrics
    st.subheader("Metrik Evaluasi")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{mae:.2f} m³")
    with col2:
        st.metric("MSE", f"{mse:.2f} m³")
    with col3:
        st.metric("R² Score", f"{r2:.2f}")
    
    # Actual vs Predicted plot
    st.subheader("Perbandingan Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y, predictions)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Aktual')
    ax.set_ylabel('Prediksi')
    st.pyplot(fig)

# Main app
def main():
    if page == "Prediksi":
        st.title("Prediksi Kapasitas Air untuk Irigasi")
        st.write("Aplikasi ini memprediksi kebutuhan kapasitas air berdasarkan parameter berikut:")
    
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            soil_moisture = st.number_input("Kelembapan Tanah (%)", min_value=0.0, max_value=100.0, value=50.0)
        
        with col2:
            land_area = st.number_input("Luas Lahan (ha)", min_value=0.0, value=1.0)
        
        with col3:
            rainfall = st.number_input("Curah Hujan (mm)", min_value=0.0, value=100.0)
    
        # Load model and scaler
        model = load_model()
        scaler = load_scaler()
        
        # Prediction button
        if st.button("Prediksi Kapasitas Air"):
            try:
                # Prepare input
                input_data = np.array([[soil_moisture, land_area, rainfall]])
                scaled_input = scaler.transform(input_data)
                input_tensor = torch.FloatTensor(scaled_input)
                
                # Make prediction
                with torch.no_grad():
                    prediction = model(input_tensor).item()
                
                # Display result
                st.success(f"Prediksi Kapasitas Air yang Dibutuhkan: {prediction:.2f} m³")
                
                # Show prediction details
                with st.expander("Detail Prediksi"):
                    st.write(f"Kelembapan Tanah: {soil_moisture}%")
                    st.write(f"Luas Lahan: {land_area} ha")
                    st.write(f"Curah Hujan: {rainfall} mm")
                    st.write(f"Kapasitas Air yang Dibutuhkan: {prediction:.2f} m³")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")
    
    elif page == "Visualisasi Data":
        visualization_page()
        
    elif page == "Evaluasi Model":
        evaluation_page()

if __name__ == "__main__":
    main()
