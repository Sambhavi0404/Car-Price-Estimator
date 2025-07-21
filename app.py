import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Synthetic car resale dataset generator
def generate_car_data(n_samples=200):
    np.random.seed(42)
    car_age = np.random.randint(1, 15, n_samples)        # Age in years
    mileage = np.random.randint(10000, 200000, n_samples) # Mileage in km
    
    # Simple price formula: Newer + less driven = higher price
    price = 30000 - (car_age * 1200) - (mileage * 0.05) + np.random.normal(0, 2000, n_samples)
    return pd.DataFrame({'car_age': car_age, 'mileage': mileage, 'price': price})

# Train a regression model
def train_model():
    df = generate_car_data()
    X = df[['car_age', 'mileage']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Car Price Estimator")
    st.write("Enter car details to estimate resale price")
    
    # Train model
    model = train_model()
    
    # Sidebar inputs
    car_age = st.slider("Car Age (years)", min_value=1, max_value=15, value=5)
    mileage = st.slider("Mileage (km)", min_value=10000, max_value=200000, step=5000, value=50000)
    
    if st.button("Predict Price"):
        prediction = model.predict([[car_age, mileage]])
        st.success(f"Estimated resale price: ${prediction[0]:,.2f}")
        
        # Visualization
        df = generate_car_data()
        fig = px.scatter_3d(df, x='car_age', y='mileage', z='price', 
                            color='price', title='Car Age & Mileage vs Price')
        fig.add_scatter3d(x=[car_age], y=[mileage], z=[prediction[0]], 
                          mode='markers', marker=dict(size=8, color='red'), name='Your Car')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
