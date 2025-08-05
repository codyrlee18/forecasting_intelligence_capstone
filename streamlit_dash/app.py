import streamlit as st
import pandas as pd
from PIL import Image
import datetime
from datetime import timedelta

forecast_table = pd.read_csv("/Users/codyrlee/Documents/CodyLee_CapstoneProject/outputs/forecast_table.csv")
high_risk_customers = pd.read_csv("/Users/codyrlee/Documents/CodyLee_CapstoneProject/outputs/high_risk_customers.csv")
KDA = pd.read_csv("/Users/codyrlee/Documents/CodyLee_CapstoneProject/outputs/KDA_output.csv")
forecast_plot = Image.open("/Users/codyrlee/Documents/CodyLee_CapstoneProject/outputs/forecast_plot.png")
kda_plot = Image.open("/Users/codyrlee/Documents/CodyLee_CapstoneProject/outputs/kda_plot.png")
data = pd.read_csv("/Users/codyrlee/Documents/CodyLee_CapstoneProject/clean_data/revenue_forecast_data.csv")

latest_month = data["month"].max()
active_latest = data[(data["month"] == latest_month) & (data['churn']== 1)]['account_id'].nunique()

data['closed_won_usd'] = data['closed_won_usd'].apply(lambda x:f"${x:,.2f}")
churned_revenue_lm = data[(data["month"] == latest_month) & (data['churn']== 1)]['closed_won_usd'].sum()

st.set_page_config(layout="wide")

st.title('Churn & Revenue Intelligence')
st.write('Welcome to the Churn & Revenue Intelligence dashboard, built to help stakeholders understand customer retention trends and revenue forecasts.')

kpi1, kpi2 = st.columns(2)

kpi1.metric(
    label = "Last Month Churn Count",
    value = f"{active_latest}"
)

kpi2.metric(
    label = "$ Churned Last Month",
    value = f"{churned_revenue_lm}"
)


st.header("Key Driver Analysis")
st.write('The following factors are the most impactful account attributes in influencing customer churn.')

kda_col1, spacer, kda_col2 = st.columns([1, 0.2, 2])

with kda_col1:
    st.dataframe(KDA)

with kda_col2:
    st.image(kda_plot, caption= "Top 5 Account Attributes in Predicting Churn")


st.header("High Risk Customers")
st.write('These customers require high-touch, white-glove attention from the CX team, as they are at high risk of churning.')
st.dataframe(high_risk_customers)

st.header("Risk-adjusted Forecast")
st.write('These are the projected next 6 months for company revenue in USD. These figures take into account churn rate.')

fc_col1, spacer, fc_col2 = st.columns([1, 0.2, 2])

with fc_col1:
    st.dataframe(forecast_table)

with fc_col2:
    st.image(forecast_plot, caption= "ARIMA Forecast Model")


