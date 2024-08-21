import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PortfolioStockAnalysis:
    expected_columns = {
        "S.N":"int64",
        "Scrip": "object",
        "Current Balance": "int64",
        "Previous Closing Price": "int64",
        "Value as of Previous Closing Price": "int64",
        "Last Transaction Price (LTP)": "int64",
        "Value as of LTP":"int64"
    }

    def __init__(self, dataframe):
        self.newdf = dataframe
        self.convert_dtypes()

    def convert_dtypes(self):
        """Convert columns to the expected data types."""
        for col, dtype in PortfolioStockAnalysis.expected_columns.items():
            if col in self.newdf.columns:
                self.newdf[col] = self.newdf[col].astype(dtype, errors='ignore')

    def validate_schema(self):
        """Validate if the dataframe has the required columns with the correct data types."""
        missing_columns = [col for col in PortfolioStockAnalysis.expected_columns if col not in self.newdf.columns]
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        for col, dtype in PortfolioStockAnalysis.expected_columns.items():
            if self.newdf[col].dtype != dtype:
                return False, f"Column '{col}' does not match the expected data type '{dtype}'."
        
        return True, None
    
    def preprocess_portfoliostock(self):
        """Preprocess the data (you can add more logic here as needed)."""
        # Additional preprocessing steps can be added here
        df = self.newdf
        script_data = pd.read_excel('Pages\\Data_Analysis\\Stock\\Script_Market_Cap.xlsx')
        #st.write(script_data)

        #st.write(df)

        ## Join the user uploaded dataset with script market cap dataset
        merged_df = pd.merge(df,script_data,on="Scrip",how="left")
        st.write(merged_df)


        return True
    def perform_profolioeda(self):
        """Perform exploratory data analysis (EDA) on the dataset."""

        return True

    def show(self):
        """Display the analysis after validation."""
        valid, message = self.validate_schema()
        if valid:
            ## Check if Data is pre Processed or not
            if PortfolioStockAnalysis.preprocess_portfoliostock(self):
                self.perform_profolioeda()
            # Additional analysis logic can be added here
        else:
            st.sidebar.error(f"Schema does not match. {message}")
            data = [
            {"Field": "S.N", "Description": "Serial Number."},
            {"Field": "Scrip", "Description": "The stock ticker symbol or script name."},
            {"Field": "Current Balance", "Description": "Total unit of particular scrip in your portfolio."},
            {"Field": "Previous Closing Price", "Description": "The previous closing price of the stock."},
            {"Field": "Value as as Previous Closing Price", "Description": "The total value of the stock based on last closing price."},
            {"Field": "Last Transaction Price (LTP)", "Description": "The last transaction price of the stock."},
            {"Field": "Value as of LTP", "Description": "The total value of the stock based on latest closing price."}
            ]

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(data)
            st.write("Your Dataset must contain following columns with strict naming convention in order to proceed")
            # Display the DataFrame as a table in Streamlit
            st.table(df.style.hide(axis="index"))