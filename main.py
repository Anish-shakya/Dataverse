import streamlit as st
import pandas as pd

# Page imports
from Pages.Data_Analysis.Sales_Analysis import SalesAnalysis
from Pages.Data_Analysis.Portfolio_Analysis import PortfolioStockAnalysis
from Pages.Data_Analysis.NepseAlphachart import NepseAlphaChart
from Pages.Data_Visualization.Visualization import Visualization
from Pages.Prediction.Global_Stock_Prediction import GlobalStockPrediction
from Pages.Prediction.Nepse_Stock_Prediction import NepseStockPrediction
from Pages.Data_Verse import About_Dataverse

class Dataverse:
    def __init__(self):
        self.setup_page()

    def setup_page(self):
        # Page configuration
        st.set_page_config(page_title="Data Verse", layout="wide")
        
        # Sidebar navigation
        st.sidebar.title("Data Verse")
        section = st.sidebar.radio("Navigation Menu", ["About Dataverse", "Analysis", "Visualization", "Prediction"])

        # Display content based on the selected section
        if section == "About Dataverse":
            About_Dataverse.home_show()

        elif section == "Analysis":
            self.analysis_section()

        elif section == "Visualization":
            self.visualization_section()

        elif section == "Prediction":
            self.prediction_section()

    def analysis_section(self):
        st.sidebar.subheader("Analysis Pages")
        dataset_title = st.sidebar.selectbox("Choose dataset type", ["Sales Analysis", "Stock Analysis"])

        if dataset_title == "Sales Analysis":
            self.sales_analysis()
        elif dataset_title == "Stock Analysis":
            self.stock_analysis()

    def sales_analysis(self):
        st.header("Sales Analysis 📊")
        uploaded_file = st.file_uploader("Upload your pharmaceutical sales dataset (CSV or XLSX)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            analysis = SalesAnalysis(df)
            analysis.show()
        else:
            st.info("Please upload a CSV file to proceed with the analysis.")

    def stock_analysis(self):
        analysis_type = st.sidebar.selectbox("Choose Analysis type", ["Portfolio Analysis", "Nespse Alpha Chart"])

        if analysis_type == "Portfolio Analysis":
            self.portfolio_analysis()
        elif analysis_type == "Nespse Alpha Chart":
            NepseAlphaChart.showchart()

    def portfolio_analysis(self):
        st.header("Portfolio Analysis 📊")
        uploaded_file = st.file_uploader("Upload your Stock dataset (CSV or XLSX)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            newdf = None
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if df.iloc[-1]['S.N'] == "Total :":
                    newdf = df.iloc[:-1]
                else:
                    newdf = df
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                if df.iloc[-1]['S.N'] == "Total :":
                    newdf = df.iloc[:-1]
                else:
                    newdf = df

            if newdf is not None:
                analysis = PortfolioStockAnalysis(newdf)
                analysis.show()
            else:
                st.error("There was an issue processing your file. Please try again.")
        else:
            st.info("Please upload a CSV or EXCEL file to proceed with the analysis.")

    def visualization_section(self):
        st.header("Data Visualization 📊")
        uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            analysis = Visualization(df)
            analysis.show()
        else:
            st.info("Please upload an EXCEL or CSV file to proceed with the visualization.")

    def prediction_section(self):
        st.sidebar.subheader("Stock Prediction Page")
        Prediction_title = st.sidebar.selectbox("Choose Market type", ["Nepse Stock Market", "Global Stock Market"])
        
        if Prediction_title == 'Nepse Stock Market':
            NepseStockPrediction.show()
        elif Prediction_title == 'Global Stock Market':
            global_app = GlobalStockPrediction()
            global_app.show()

# Run the app
if __name__ == "__main__":
    Dataverse()


    
