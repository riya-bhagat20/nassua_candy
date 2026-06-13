import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# pip install pipreqs
# xampp control panel for mysql databasei n8n
st.set_page_config(page_title='Nassau Candy Distributor | Factory Optimizer', page_icon='🍫', layout='wide') # webfx for icon
st.title('Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor')
st.markdown("""
<div class="hero-header">
    <h1>Factory Optimization Engine</h1>
    <p>ML-driven factory–product reassignment · Lead time prediction · Scenario simulation · Margin recovery</p>
</div>
""", unsafe_allow_html=True)

#  ----------------------- sidebar --------------------------


st.sidebar.image("images1.png", width=200)

st.sidebar.header('Nassau Candy Distributor | Factory Optimizer')
upload_file = st.sidebar.file_uploader("Upload Dataset (CSV / Excel )", type=["csv", "xlsx", "xls"])

if upload_file is not None:
    if upload_file.name.endswith('.csv'):
        df = pd.read_csv(upload_file)
    else:
        df = pd.read_excel(upload_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())
    st.write("Dataset Shape:", df.shape)
elif upload_file is None:
    df=pd.read_csv("Nassau Candy Distributor.csv")    
else:
    st.warning("Please upload a dataset to proceed.")

# ------------------------------- data cleaning--------------------------

# ========find duplicate and null values========

duplicate_rows = df[df.duplicated()]

# st.write("Duplicate Rows:")
# st.dataframe(duplicate_rows)

null_values = df.isnull().sum()

# st.write("Null Values in Each Column:")
# st.dataframe(null_values)

# -----------------------convert date in to datetime--------------------------

df['Customer ID'] = df['Customer ID'].astype(int)
df['Order Date'] = pd.to_datetime(df['Order Date'],errors='coerce',format='mixed',dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'],errors='coerce',format='mixed',dayfirst=True)
df['profit_margin'] = df['Gross Profit'] / df['Sales']
df['Lead Time (Days)'] = (df['Ship Date'] - df['Order Date']).dt.days
df["Lead Time (Days)"] = df["Lead Time (Days)"].clip(lower=0)
df['month'] = df['Order Date'].dt.month
df['year'] = df['Order Date'].dt.year

# --------------------------- remove outliers--------------------------

for col in ['Sales', 'Cost', 'Gross Profit','Units','Lead Time (Days)']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col]=df[col].clip(lower=lower_bound, upper=upper_bound, inplace=True)


    




# -----------------add filter---------------------

st.sidebar.header('Filter Data')
RESION=st.sidebar.multiselect('Select Region', options=df['Country/Region'].unique(), default=df['Country/Region'].unique())   
SHIP_MODE=st.sidebar.multiselect('Select Ship Mode', options=df['Ship Mode'].unique(), default=df['Ship Mode'].unique())
DIVISION=st.sidebar.multiselect('Select Division', options=df['Division'].unique(), default=df['Division'].unique())

#---------------- connect filder to data on the base categorical data----------------------

df_selection=df.query("`Country/Region` == @RESION & `Ship Mode` == @SHIP_MODE & Division == @DIVISION")

st.markdown("---")


# ------------------------------------ overview ----------------------------------- 

def overview_page():
    st.header("Overview") 
    Total_Orders=float(df_selection['Units'].count())
    Total_Sales=float(df_selection['Sales'].sum())
    Total_Cost=float(df_selection['Cost'].sum())
    Total_Profit=float(df_selection['Gross Profit'].sum())    


    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="large")

    kpi1.metric(label="Total Orders", value=f"{Total_Orders:,.0f}")
    kpi2.metric(label="Total Sales", value=f"{Total_Sales:,.2f}", delta=None)
    kpi3.metric(label="Total Cost", value=f"{Total_Cost:,.2f}", delta=None)
    kpi4.metric(label="Total Profit", value=f"{Total_Profit:,.2f}", delta=None)
    st.markdown("---")


# ------------------------------ sales by region and ship mode bar & pie chart --------------------------


    col1, col2 = st.columns(2, gap="large")

    with col1:
        
        sales_by_region = df_selection.groupby('Region')[['Sales','Gross Profit']].sum().reset_index()
        fig1 = px.bar(sales_by_region, x='Region', y=['Sales','Gross Profit'], 
                      barmode='group',color_discrete_sequence=["#C27B2E","#2EC27B"],
                      title='Total Sales & Gross Profit by Region')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        sales_by_ship_mode = df_selection.groupby('Ship Mode')['Sales'].sum().reset_index()
        fig2 = px.pie(sales_by_ship_mode, values='Sales', names='Ship Mode', title='Total Sales by Ship Mode')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

# ------------------------------ sales by product bar chart & expander--------------------------

    col3, col4 = st.columns(2, gap="large")

    with col3:
        
        sales_by_Product = df_selection.groupby('Region')['Sales'].sum().sort_values(ascending=True).head(5).reset_index()
        fig3 = px.bar(sales_by_Product,x='Region', y='Sales', orientation='v',
                      color_continuous_scale=["#3B1A08","#C27B2E","#FDF6EC"],
                      title='Total Sales by Product')
        
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        #---------------------- create expander----------------------

        expander=st.expander('Top 5 Products by Sales')
        data=df[['Product Name','Sales']].groupby(by='Product Name')['Sales'].sum().sort_values(ascending=False).head(5).reset_index()  
        expander.write(data)


    with col4:
        if "Product Name" in df_selection.columns:
            prod = df_selection.groupby("Product Name")["Sales"].sum().sort_values(ascending=True).tail(10).reset_index()
            fig4 = px.bar(prod, x="Sales", y="Product Name", orientation="h",
                          title="Sales by Product", color="Sales",
                          color_continuous_scale=["#FDF6EC","#C27B2E","#3B1A08"])
            
            fig4.update_coloraxes(showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

            expander=st.expander('Top 10 Products by Sales')
            data=df[['Product Name','Sales']].groupby(by='Product Name')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()  
            expander.write(data)

    st.markdown("---")

    with st.expander("Dataset Preview"):  
        st.dataframe(df_selection).head(20)
# ---------------------------- EDA --------------------------

def EDA_page():
    st.header("Exploratory Data Analysis")
    st.write("This section will contain various EDA visualizations and insights based on the uploaded dataset.")

    # Example EDA visualization

    fig = px.histogram(df, x='Sales', nbins=30, title='Distribution of Sales')
    st.plotly_chart(fig)

    st.write("Dataset Description:")
    st.dataframe(df.describe())
    

    info_df = pd.DataFrame({
    "Column": df.columns,
    "Non-Null Count": df.count().values,
    "Dtype": df.dtypes.values
    })

    st.dataframe(info_df)

    

    


#  ---------------------------- Navigation --------------------------

nav_options = st.sidebar.radio('Navigation',["Overview", "Exploratory Data Analysis", "ML Model", "Clustering","Scenario Simulation","Recommendations"])

if nav_options == "Overview":
  
    overview_page()

elif nav_options == "Exploratory Data Analysis":
    EDA_page()