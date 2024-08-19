import streamlit as st
import pygwalker as pyg
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
class Visualization:
    def __init__(self, dataframe=None):
        self.df = dataframe

    def show(self):
        st.title("Data Visualization")
        st.write("Visualize your data as per your need with DataVerse") 
        self.load_dashboard()
        

    def load_dashboard(self):
        st.sidebar.subheader("Upload Dataset")
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # Determine file type and read into DataFrame
            if uploaded_file.name.endswith(".csv"):
                self.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                self.df = pd.read_excel(uploaded_file)

            # Show a sample of the uploaded data
            st.write("Data Preview:")
            st.write(self.df.head())

            # Add more visualization logic here, e.g., using PYGwalker
            if self.df is not None:
                dashboard = StreamlitRenderer(self.df)
                dashboard.explorer()
                  # Replace this with your actual PYGwalker integration

  
