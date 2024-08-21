import streamlit as st
import pandas as pd

# Page imports
#from Pages.Data_Analysis.Sales_Analysis import SalesAnalysis
from Pages.Data_Analysis.Pharmaceutical_Sales_Analysis import PharmaceuticalSalesAnalysis
#from Pages.Data_Analysis.Stock_Analysis import StockAnalysis
from Pages.Data_Analysis.NepseStock_Analysis import NepseStockAnalysis
from Pages.Data_Analysis.Portfolio_Analysis import PortfolioStockAnalysis
from Pages.Data_Analysis.Describe import Describe
from Pages.Data_Visualization.Visualization import Visualization
from Pages.Prediction.Stock_Trend_Prediction import StockTrendPrediction
from Pages.Recommendation.Movie_Recommendation import MovieRecommendation
from Pages.Recommendation.Book_Recommendation import BookRecommendation
from Pages.Data_Verse import About_Dataverse

## app configuration
st.set_page_config(page_title="Data Verse",layout="wide")
# Sidebar navigation
st.sidebar.title("Data Verse")
section = st.sidebar.radio("Navigation Menu", ["About Dataverse", "Analysis", "Visualization", "Prediction", "Recommendation"])



# Display content based on the selected section
if section == "About Dataverse":
    About_Dataverse.home_show()

elif section == "Analysis":
    st.sidebar.subheader("Analysis Pages")
    dataset_title = st.sidebar.selectbox("Choose dataset type", ["Pharmaceutical Sales","Sales Analysis", "Stock Analysis", "Describe Dataset"])

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
        analysis_type = st.sidebar.selectbox("Choose Analysis type", ["Nepase Stock Analysis", "Portfolio Analysis"])

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
                st.info("Please upload a CSV file to proceed with the analysis.")

        elif analysis_type =="Portfolio Analysis":
            st.header("Portfoli Analysis 📊")
            uploaded_file = st.file_uploader("Upload your Stock dataset (CSV or XLSX)", type=["csv","xlsx"])
            
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    df=pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df=pd.read_excel(uploaded_file)
                analysis = PortfolioStockAnalysis(df)  # Instantiate the class with the DataFrame
                analysis.show()  # Call the show method to perform analysis
            else:
                st.info("Please upload a CSV file to proceed with the analysis.")


    # uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    # if uploaded_file:
    #     if uploaded_file.name.endswith(".csv"):
    #         df = pd.read_csv(uploaded_file)
    #     elif uploaded_file.name.endswith(".xlsx"):
    #         df = pd.read_excel(uploaded_file)

    #     if dataset_title == "Sales Dataset":
    #         analysis = SalesAnalysis(df)
    #         analysis.show()
    #     elif dataset_title == "Pharmaceutical Sales":
    #         analysis = PharmaceuticalSalesAnalysis(df)
    #         analysis.show()
    #     elif dataset_title == "Stock Dataset":
    #         analysis = StockAnalysis(df)
    #         analysis.show()
    #     elif dataset_title == "Describe Dataset":
    #         Describe.show()
        
elif section == "Visualization":
    viz= Visualization()
    viz.show()

elif section == "Prediction":
    StockTrendPrediction.show()

elif section == "Recommendation":
    st.sidebar.subheader("Recommendation Pages")
    page = st.sidebar.radio("Select a page:", ["Movie Recommendation", "Book Recommendation"])
    if page == "Movie Recommendation":
        MovieRecommendation.show()
    elif page == "Book Recommendation":
        BookRecommendation.show()
