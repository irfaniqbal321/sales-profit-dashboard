# Sales Performance Analytics and Profit Prediction

## Business Problem
Company wants to identify low-profit categories and predict future profit.

## Dataset Features
1500 orders with Customer, Product, Sales details.

## Feature Engineering Done
- Profit_Margin = Profit/Total_Sale*100
- Age_Group from Customer_Age
- Order_Month from Order_Date
- Total_Sale and Profit calculated

## Dashboard Pages
1. Home - KPIs
2. Sales Dashboard - Category & Time analysis
3. Product Analysis - Discount vs Profit
4. Profit Prediction - User input prediction
5. Model Performance - Random Forest R2=0.996

## Model
RandomForestRegressor with features: Quantity, Unit_Price, Discount, Shipping_Cost

## How to Run
pip install -r requirements.txt
streamlit run app.py