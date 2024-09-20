import streamlit as st
import pandas as pd

# Page imports
#from Pages.Data_Analysis.Sales_Analysis import SalesAnalysis
from Pages.Data_Analysis.Pharmaceutical_Sales_Analysis import PharmaceuticalSalesAnalysis
#from Pages.Data_Analysis.Stock_Analysis import StockAnalysis
from Pages.Data_Analysis.NepseStock_Analysis import NepseStockAnalysis
from Pages.Data_Analysis.Portfolio_Analysis import PortfolioStockAnalysis
from Pages.Data_Analysis.NepseAlphachart import NepseAlphaChart
from Pages.Data_Analysis.Describe import Describe
from Pages.Data_Visualization.Visualization import Visualization
from Pages.Prediction.Global_Stock_Prediction import GlobalStockPrediction
from Pages.Prediction.Nepse_Stock_Prediction import NepseStockPrediction
from Pages.Recommendation.Stock_Recommendation import StockRecommendation
from Pages.Recommendation.Sales_Stock_Recommendation import SalesRestockRecommendation
from Pages.Data_Verse import About_Dataverse

## app configuration
st.set_page_config(page_title="Data Verse",layout="wide")
# Sidebar navigation
st.sidebar.title("Data Verse")
section = st.sidebar.radio("Navigation Menu", ["About Dataverse", "Analysis", "Visualization", "Prediction"])



# Display content based on the selected section
if section == "About Dataverse":
    About_Dataverse.home_show()

elif section == "Analysis":
    st.sidebar.subheader("Analysis Pages")
    dataset_title = st.sidebar.selectbox("Choose dataset type", ["Pharmaceutical Sales", "Stock Analysis", "Describe Dataset"])

    if dataset_title == "Pharmaceutical Sales":

        st.header("Pharmaceutical Sales Analysis 📊")
        uploaded_file = st.file_uploader("Upload your pharmaceutical sales dataset (CSV or XLSX)", type=["csv","xlsx"])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df=pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df=pd.read_excel(uploaded_file)
            analysis = PharmaceuticalSalesAnalysis(df)  # Instantiate the class with the DataFrame
            analysis.show()  # Call the show method to perform analysis
        else:
            st.info("Please upload a CSV file to proceed with the analysis.")

    elif dataset_title == "Stock Analysis":
        analysis_type = st.sidebar.selectbox("Choose Analysis type", ["Portfolio Analysis","Nepase Stock Analysis", "Nespse Alpha Chart"])

        if analysis_type == "Nepase Stock Analysis":
            st.header("Nepse Stock Analysis 📊")
            uploaded_file = st.file_uploader("Upload your Stock dataset (CSV or XLSX)", type=["csv","xlsx"])
            
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    df=pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df=pd.read_excel(uploaded_file)
                analysis = NepseStockAnalysis(df)  # Instantiate the class with the DataFrame
                analysis.show()  # Call the show method to perform analysis
            else:
                st.info("Please upload a EXCEL or CSV file to proceed with the analysis.")
            
        elif analysis_type =="Portfolio Analysis":
            st.header("Portfolio Analysis 📊")
            uploaded_file = st.file_uploader("Upload your Stock dataset (CSV or XLSX)", type=["csv","xlsx"])
            
            if uploaded_file is not None:
                newdf =None
                if uploaded_file.name.endswith('.csv'):
                    df=pd.read_csv(uploaded_file)
                    if df.iloc[-1]['S.N'] == "Total :":
                        newdf=df.iloc[:-1]
                    else:
                        newdf =df
                        
                elif uploaded_file.name.endswith('.xlsx'):
                    df=pd.read_excel(uploaded_file)
                    if df.iloc[-1]['S.N'] == "Total :":
                        newdf=df.iloc[:-1]
                    else:
                        df=newdf

                if newdf is not None:
                    analysis = PortfolioStockAnalysis(newdf)  # Instantiate the class with the DataFrame
                    analysis.show()  # Call the show method to perform analysis
                else:
                    st.error("There was an issue processing your file. Please try again.")

                # analysis = PortfolioStockAnalysis(newdf)  # Instantiate the class with the DataFrame
                # analysis.show()  # Call the show method to perform analysis

            else:
                st.info("Please upload a CSV or EXCEL file to proceed with the analysis.")

        elif analysis_type == "Nespse Alpha Chart":
            NepseAlphaChart.showchart()
        
elif section == "Visualization":
    st.header("Data Visualization 📊")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv","xlsx"])
            
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df=pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df=pd.read_excel(uploaded_file)
            analysis = Visualization(df)  # Instantiate the class with the DataFrame
            analysis.show() # Call the show method to perform analysis
    else:
        st.info("Please upload an EXCEL or CSV file to proceed with the visualization.")
    
elif section == "Prediction":
    st.sidebar.subheader("Stock Prediction Page")
    Prediction_title = st.sidebar.selectbox("Choose Market type", ["Nepse Stock Market", "Global Stock Market"])
    if Prediction_title == 'Nepse Stock Market':
        NepseStockPrediction.show()
    elif Prediction_title == 'Global Stock Market':
        GlobalStockPrediction.show()

# elif section == "Recommendation":
#     st.sidebar.subheader("Recommendation Pages")
#     page = st.sidebar.radio("Select a page:", ["Sales Restock Recommendation", "Stock Recommendation"])
#     if page == "Sales Restock Recommendation":
#         SalesRestockRecommendation.show()
#     elif page == "Stock Recommendation":
#         StockRecommendation.show() 
    
