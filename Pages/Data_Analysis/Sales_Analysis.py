import streamlit as st
import pandas as pd
import numpy as np

class SalesAnalysis:
    expected_column = {
    "ID": "int64",
    "Date": "datetime64[ns]",
    "Product": "object",
    "Cost_Price": "float64",
    "Selling_Price": "float64",
    "Quantity": "int64",
    "Amount": "float64",
    "Discounts": "float64",
    "Net Amount": "float64",
    "Payment Mode":"object"
    }


    def __init__(self, dataframe):
        self.df = dataframe
        self.convert_dtypes()

    def convert_dtypes(self):
        for col, dtype in SalesAnalysis.expected_schema.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype, errors='ignore')

    # def validate_schema(self):
    #     if list(self.df.columns) != list(SalesAnalysis.expected_schema.keys()):
    #         return False, "Column names do not match the expected schema."
        
    #     for col, dtype in SalesAnalysis.expected_schema.items():
    #         if self.df[col].dtype != dtype:
    #             return False, f"Column '{col}' does not match the expected data type '{dtype}'."
    #     return True, None
    
    def preprocess(self):


        return True

    def perform_eda(self):
        df = self.df
        st.write(df)

    def show(self):
        valid, message = self.validate_schema()
        if valid:
            st.title("Sales Analysis")
            self.perform_eda()
            # Additional sales analysis logic here
        else:
            st.sidebar.error(f"Schema does not match. {message}")
