# 📊 Sales Performance Analytics & Profit Prediction

> A complete Business Intelligence solution to identify low-profit categories and predict future profit.

🔗 **Live Demo:** [https://irfan-iqbal-sales-dashboard.streamlit.app]  
🔗 **GitHub:** https://github.com/irfaniqbal321/sales-profit-dashboard

## 🎯 Business Problem
Company wants to identify low-profit categories and predict future profit to optimize business strategy.

## 📂 Dataset Features
- **1500 Orders** with Customer, Product, Sales details
- Features: Customer_Age, Order_Date, Quantity, Unit_Price, Discount, Shipping_Cost

## ⚙️ Feature Engineering Done
- `Profit_Margin = Profit/Total_Sale*100`
- `Age_Group` from Customer_Age
- `Order_Month` from Order_Date
- `Total_Sale` and `Profit` calculated fields

## 📑 Dashboard Pages
1.  **Home** - KPIs (Total Sales, Profit, Customers)
2.  **Sales Dashboard** - Category & Time analysis
3.  **Product Analysis** - Discount vs Profit insights
4.  **Profit Prediction** - User input based prediction
5.  **Model Performance** - Random Forest R² = 0.996

## 🤖 ML Model
- **Model:** RandomForestRegressor
- **Features:** Quantity, Unit_Price, Discount, Shipping_Cost
- **Performance:** R² Score 0.996
- **Target:** Profit Prediction

## 🛠️ Tech Stack
Python, Streamlit, Pandas, Scikit-learn, Plotly

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py