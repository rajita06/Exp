import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data: Expense data
data = {
    'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'Food': [300, 320, 280, 350, 360, 400, 420, 450, 500, 550, 600, 620],
    'Entertainment': [150, 130, 170, 160, 180, 190, 210, 220, 250, 270, 300, 320],
    'Utilities': [100, 110, 90, 120, 130, 140, 150, 160, 180, 190, 200, 210],
    'Total_Expense': [550, 560, 540, 630, 670, 730, 780, 830, 930, 1010, 1100, 1150]
}

# Create DataFrame
df = pd.DataFrame(data)

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Food_Share'] = df['Food'] / df['Total_Expense']
df['Entertainment_Share'] = df['Entertainment'] / df['Total_Expense']
df['Utilities_Share'] = df['Utilities'] / df['Total_Expense']

# Features and Target
X = df[['Food', 'Entertainment', 'Utilities', 'Food_Share', 'Entertainment_Share', 'Utilities_Share']]
y = df['Total_Expense']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Expense Prediction and Visualization App")

# Buttons for actions
if st.button("Predict Total Expenses"):
    y_pred = model.predict(X_test)
    st.write("### Model Evaluation")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Predicted vs Actual Visualization
    st.write("### Predicted vs Actual Total Expenses")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    ax.set_xlabel("Actual Total Expenses")
    ax.set_ylabel("Predicted Total Expenses")
    st.pyplot(fig)

if st.button("Show Visualizations"):
    st.write("### Monthly Spending Trends")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Date', y='Total_Expense', data=df, label='Total Expenses', ax=ax)
    sns.lineplot(x='Date', y='Food', data=df, label='Food', ax=ax)
    sns.lineplot(x='Date', y='Entertainment', data=df, label='Entertainment', ax=ax)
    sns.lineplot(x='Date', y='Utilities', data=df, label='Utilities', ax=ax)
    ax.set_title("Monthly Spending Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    ax.legend()
    st.pyplot(fig)

st.write("### Suggested Budgeting Goals")
df['Predicted_Expense'] = model.predict(X)
df['Budget_Goal'] = df['Predicted_Expense'] * 1.05  # 5% more than predicted
st.dataframe(df[['Date', 'Total_Expense', 'Predicted_Expense', 'Budget_Goal']])
