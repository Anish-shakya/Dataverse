import streamlit as st
import pandas as pd
import numpy as no
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as df
import yfinance as yf
from datetime import date,timedelta

import plotly.graph_objects as go
import plotly.express as px

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

class GlobalStockPrediction:

    def globalprediction():
        st.sidebar.header('Select the parameters from below')
        start_date  = st.sidebar.date_input('Start Date',date(2020,1,1))
        end_date = st.sidebar.date_input('End Date',date.today())

        ## add ticker symbol list 
        ticker_list = ['AAPL','GOOGL','META','MSFT','GOOG','TSLA','NVDA','ADBE','PYPL','INTC','CMCSA','NFLX','PEP']
        ticker = st.sidebar.selectbox('Select Stock',ticker_list)


        ### Fetch Data from user input form yfinance
        data = yf.download(ticker,start=start_date,end=end_date)

        data.insert(0,'Date',data.index,True)
        data.reset_index(drop=True,inplace=True)

        st.write('Data from',start_date ,'to ',end_date)
        st.write(data.head())

        #plot the data
        st.header( 'Data Visualization')
        st.write('**Note**: Select your specific date range on the sidebar, or zoom in on the plot and select your specific column')
        fig = px.line(data,x='Date',y=data.columns, title='Closing Price of the Stock', width=2000,height=600)
        st.plotly_chart(fig)

        ### add a select box to select column from data
        column = st.selectbox('Select the column to be used for forcasting',data.columns[1:])

        ## Subsetting the data
        data = data[['Date',column]]
        st.write("Selected Data")
        st.write(data)

        ### ADF rest check Stationary
        st.header('Is the Data Stationary?')
        st.write('**Note:** If p-value is less than 00.5, then data is stationary')
        st.write(adfuller(data[column])[1] <0.05)
        
        ### Lets Decompose the Data
        st.header('Decomposition of the data')
        decomposition = seasonal_decompose(data[column],model='additive',period=12)
        st.plotly_chart(px.line(x=data['Date'],y=decomposition.trend,title='Trend',width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
        st.plotly_chart(px.line(x=data['Date'],y=decomposition.seasonal,title='Seasonality',width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
        st.plotly_chart(px.line(x=data['Date'],y=decomposition.resid,title='Residuals',width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Red',line_dash='dot'))

        ## user input for three parameters of the model abd seasonal order
        p = st.slider('Select the value of p',0, 5, 2)
        d = st.slider('Select the value of d',0, 5, 1)
        q = st.slider('Select the value of q',0, 5, 2)
        seasonal_order = st.number_input('select the value of seasonal p',0, 24, 12)
    def show():
        st.header("Global Stock Trend Prediction Using SARIMA Model 📊")
        st.subheader("This is app is created to forcaste the stock market price of the selected company")
        GlobalStockPrediction.globalprediction()
        # Additional prediction logic here it goes
