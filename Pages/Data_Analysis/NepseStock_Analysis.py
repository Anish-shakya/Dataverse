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

class NepseStockAnalysis:
    expected_columns = {
        "Date": "datetime64[ns]",
        "Script": "object",
        "Open": "int64",
        "Close": "int64",
        "High": "int64",
        "Low": "int64",

    }
    def __init__(self, dataframe):
        self.df = dataframe
        self.convert_dtypes()

    def convert_dtypes(self):
        """Convert columns to the expected data types."""
        for col, dtype in NepseStockAnalysis.expected_columns.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype, errors='ignore')

    def validate_schema(self):
        """Validate if the dataframe has the required columns with the correct data types."""
        missing_columns = [col for col in NepseStockAnalysis.expected_columns if col not in self.df.columns]
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        for col, dtype in NepseStockAnalysis.expected_columns.items():
            if self.df[col].dtype != dtype:
                return False, f"Column '{col}' does not match the expected data type '{dtype}'."
        
        return True, None
    
    def preprocess_stock(self):
        """Preprocess the data (you can add more logic here as needed)."""
        # Additional preprocessing steps can be added here

        return True
    def perform_stockeda(self):
        """Perform exploratory data analysis (EDA) on the dataset."""
        df = self.df
        st.write(df)

    def show(self):
        """Display the analysis after validation."""
        valid, message = self.validate_schema()
        if valid:
            ## Check if Data is pre Processed or not
            if NepseStockAnalysis.preprocess_stock(self):
                self.perform_stockeda()
            # Additional analysis logic can be added here
        else:
            st.sidebar.error(f"Schema does not match. {message}")
            data = [
            {"Field": "Date", "Description": "	The date of the stock traded."},
            {"Field": "Script", "Description": "The stock ticker symbol or script name."},
            {"Field": "Open", "Description": "The opening price of the stock for the given date."},
            {"Field": "Close", "Description": "The closing price of the stock for the given date."},
            {"Field": "High", "Description": "The highest price reached by the stock on that date."},
            {"Field": "Low", "Description": "The lowest price reached by the stock on that date."}
            ]

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(data)
            st.write("Your Dataset must contain following columns with strict naming convention in order to proceed")
            # Display the DataFrame as a table in Streamlit
            st.table(df.style.hide(axis="index"))