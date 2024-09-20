import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

class GlobalStockPrediction:

    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.ticker = None
        self.data = None
        self.column = None
        self.model = None
        self.forecast_period = None
        self.predictions = None

    def user_input(self):
        """Get the date range and stock ticker symbol from the user."""
        st.sidebar.header('Select the parameters from below')
        self.start_date = st.sidebar.date_input('Start Date', date(2020, 1, 1))
        self.end_date = st.sidebar.date_input('End Date', date.today())
        ticker_list = ['AAPL', 'GOOGL', 'META', 'MSFT', 'GOOG', 'TSLA', 'NVDA', 'ADBE', 'PYPL', 'INTC', 'CMCSA', 'NFLX', 'PEP']
        self.ticker = st.sidebar.selectbox('Select Stock', ticker_list)

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance based on user input."""
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data.insert(0, 'Date', self.data.index, True)
        self.data.reset_index(drop=True, inplace=True)
        st.write(f'Data from {self.start_date} to {self.end_date}')
        st.write(self.data.head())

    def visualize_data(self):
        """Visualize the stock data."""
        st.header('Data Visualization')
        st.write('**Note**: Select your specific date range on the sidebar or zoom in on the plot and select your specific column.')
        fig = px.line(self.data, x='Date', y=self.data.columns, title='Closing Price of the Stock', width=2000, height=600)
        st.plotly_chart(fig)

    def select_column_for_forecast(self):
        """Let the user select which column to use for forecasting."""
        self.column = st.selectbox('Select the column to be used for forecasting', self.data.columns[1:])
        self.data = self.data[['Date', self.column]]
        st.write("Selected Data")
        st.write(self.data)

    def adf_test(self):
        """Perform the Augmented Dickey-Fuller test to check if data is stationary."""
        st.header('Is the Data Stationary?')
        st.write('**Note:** If p-value is less than 0.05, then the data is stationary.')
        adf_result = adfuller(self.data[self.column])[1] < 0.05
        st.write(adf_result)

    def decompose_data(self):
        """Decompose the time series data into trend, seasonality, and residuals."""
        st.header('Decomposition of the data')
        decomposition = seasonal_decompose(self.data[self.column], model='additive', period=12)
        self.plot_decomposition(decomposition)

    def plot_decomposition(self, decomposition):
        """Helper function to plot the decomposed data (trend, seasonality, residuals)."""
        st.plotly_chart(px.line(x=self.data['Date'], y=decomposition.trend, title='Trend', width=1200, height=400)
                        .update_traces(line_color='Blue'))
        st.plotly_chart(px.line(x=self.data['Date'], y=decomposition.seasonal, title='Seasonality', width=1200, height=400)
                        .update_traces(line_color='Green'))
        st.plotly_chart(px.line(x=self.data['Date'], y=decomposition.resid, title='Residuals', width=1200, height=400)
                        .update_traces(line_color='Red', line_dash='dot'))

    def user_input_model_params(self):
        """Get SARIMA model parameters from the user."""
        p = st.slider('Select the value of p', 0, 5, 2)
        d = st.slider('Select the value of d', 0, 5, 1)
        q = st.slider('Select the value of q', 0, 5, 2)
        seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)
        return p, d, q, seasonal_order

    def build_model(self, p, d, q, seasonal_order):
        """Build and fit the SARIMA model."""
        self.model = sm.tsa.statespace.SARIMAX(self.data[self.column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order)).fit()
        st.header('Model Summary')
        st.write(self.model.summary())

    def forecast(self):
        """Make predictions using the SARIMA model."""
        st.write("<p style='color:green;font-size:50px; font-weight:bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
        self.forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
        self.predictions = self.model.get_prediction(start=len(self.data), end=len(self.data) + self.forecast_period)
        self.predictions = self.predictions.predicted_mean

        self.predictions.index = pd.date_range(start=self.end_date, periods=len(self.predictions), freq='D')
        self.predictions = pd.DataFrame(self.predictions)
        self.predictions.insert(0, 'Date', self.predictions.index)
        self.predictions.reset_index(drop=True, inplace=True)

        st.write("## Predictions", self.predictions)
        st.write("## Actual Data", self.data)
        self.plot_predictions()

    def plot_predictions(self):
        """Plot actual vs predicted stock prices."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data['Date'], y=self.data[self.column], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.predictions['Date'], y=self.predictions['predicted_mean'], mode='lines', name='Predicted', line=dict(color='red')))
        fig.update_layout(title='Actual Vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
        st.plotly_chart(fig)

    def show_hide_buttons(self):
        """Show and hide individual plots using buttons."""
        show_plots = False
        if st.button('Show Separate Plots'):
            if not show_plots:
                st.write(px.line(x=self.data['Date'], y=self.data[self.column], title='Actual', width=1200, height=400)
                         .update_traces(line_color='Blue'))
                st.write(px.line(x=self.predictions['Date'], y=self.predictions['predicted_mean'], title='Predicted', width=1200, height=400)
                         .update_traces(line_color='Red'))
                show_plots = True
            else:
                show_plots = False

        hide_plots = False
        if st.button("Hide Separate Plots"):
            if not hide_plots:
                hide_plots = True
            else:
                hide_plots = False

    def show(self):
        """Main function to display the app."""
        st.header("Global Stock Trend Prediction Using SARIMA Model 📊")
        st.subheader("This app forecasts the stock market price of the selected company")
        self.user_input()
        self.fetch_data()
        self.visualize_data()
        self.select_column_for_forecast()
        self.adf_test()
        self.decompose_data()
        p, d, q, seasonal_order = self.user_input_model_params()
        self.build_model(p, d, q, seasonal_order)
        self.forecast()
        self.show_hide_buttons()
