import streamlit as st
import pygwalker as pyg
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
class Visualization:
    def __init__(self, dataframe=None):
        self.df = dataframe        

    def load_dashboard(self):
        df=self.df
        st.write("Data Preview:")
        st.write(self.df.head())

            # Add more visualization logic here, e.g., using PYGwalker
        if self.df is not None:
            dashboard = StreamlitRenderer(self.df)
            dashboard.explorer()
                  # Replace this with your actual PYGwalker integration
    def show(self):
        self.load_dashboard()

  
