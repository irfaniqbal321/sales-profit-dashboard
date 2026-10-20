import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sales Performance Analytics", layout="centered")

@st.cache_data
def load_data():
    np.random.seed(42)
    n=500
    data = {
        'Order_Date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'Customer_Gender': np.random.choice(['Male','Female'], n),
        'Customer_Age': np.random.randint(18,65,n),
        'City': np.random.choice(['Karachi','Lahore','Islamabad'], n),
        'Region': np.random.choice(['Sindh','Punjab','North'], n),
        'Product_Category': np.random.choice(['Electronics','Clothing','Furniture','Grocery'], n),
        'Quantity': np.random.randint(1,10,n),
        'Unit_Price': np.random.randint(200,5000,n),
        'Discount': np.random.choice([0,0.05,0.1,0.2], n),
        'Shipping_Cost': np.random.randint(50,250,n),
        'Customer_Rating': np.random.randint(1,6,n),
        'Returned': np.random.choice(['Yes','No'], n, p=[0.12,0.88]),
    }
    df = pd.DataFrame(data)
    df['Total_Sale'] = df['Quantity']*df['Unit_Price']*(1-df['Discount'])+df['Shipping_Cost']
    df['Profit'] = df['Total_Sale']*0.22 - df['Shipping_Cost']*0.5
    df['Profit_Margin'] = df['Profit']/df['Total_Sale']*100
    df['Order_Month'] = pd.to_datetime(df['Order_Date']).dt.month
    return df

df = load_data()

page = st.selectbox("Go to", ["Home","Sales Dashboard","Product Analysis","Profit Prediction","Model Performance"])

if page == "Home":
    st.title("Sales Performance Analytics")
    st.write("### By Irfan Iqbal - Hackathon 2026")
    col1,col2,col3 = st.columns(3)
    col1.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    col2.metric("Total Sales", f"${df['Total_Sale'].sum():,.0f}")
    col3.metric("Avg Margin", f"{df['Profit_Margin'].mean():.1f}%")
    st.write("**Business Problem:** Company wants to identify low-profit categories and predict future profit.")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Sales Dashboard":
    st.title("Sales Dashboard")
    st.subheader("Sales by Category")
    st.bar_chart(df.groupby('Product_Category')['Total_Sale'].sum())
    st.subheader("Monthly Sales Trend")
    st.line_chart(df.groupby('Order_Month')['Total_Sale'].sum())

elif page == "Product Analysis":
    st.title("Product & Customer Analysis")
    st.subheader("Discount vs Profit")
    st.scatter_chart(df, x='Discount', y='Profit', color='Product_Category')

elif page == "Profit Prediction":
    st.title("Profit Prediction")
    qty = st.number_input("Quantity",1,20,2)
    price = st.number_input("Unit Price",100,10000,1000)
    discount = st.slider("Discount",0.0,0.5,0.1)
    ship = st.number_input("Shipping Cost", 50, 500, 100)
    if st.button("Predict Profit"):
        profit = (qty*price*(1-discount)+ship)*0.22 - ship*0.5
        st.success(f"Predicted Profit: ${profit:.2f}")

elif page == "Model Performance":
    st.title("ML Model - Random Forest")
    if st.button("Train Model"):
        X = df[['Quantity','Unit_Price','Discount','Shipping_Cost']]
        y = df['Profit']
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=20).fit(X_train,y_train)
        score = model.score(X_test,y_test)
        st.metric("R2 Score", f"{score:.3f}")