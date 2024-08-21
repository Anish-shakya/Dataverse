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
        df=self.newdf
        df['Price Change (%)'] = (df['Last Transaction Price (LTP)'] - df['Previous Closing Price'])/df['Previous Closing Price'] * 100

        df['Value chage (%)'] = (df['Value as of LTP'] - df['Value as of Previous Closing Price'])/df['Value as of Previous Closing Price'] * 100

        return True
    
    def perform_profolioeda(self):
        """Perform exploratory data analysis (EDA) on the dataset."""
        df = self.newdf
        script_data = pd.read_excel('Pages\\Data_Analysis\\Stock\\Script_Market_Cap.xlsx')
        script_data = script_data.drop(columns=['S.N'],errors='ignore')
        #st.write(script_data)
        #st.write(df)
        ## Join the user uploaded dataset with script market cap dataset
        merged_df = pd.merge(df,script_data,on="Scrip",how="left")
        merged_df2 = merged_df.copy()

        # Sidebar filter for selecting scrip
        st.sidebar.header('Choose Your Filters:')
        scrip_options = merged_df2['Scrip'].unique()
        selected_scrip = st.sidebar.multiselect('Select Scrip', scrip_options)

        # Apply the filter
        if not selected_scrip:
            merged_df2 = merged_df.copy()
        else:
            merged_df2 = merged_df2[merged_df2['Scrip'].isin(selected_scrip)] 
        
        ## Filter for sectors
        sector_options = merged_df2['Sector'].unique()
        selected_sector = st.sidebar.multiselect('Select Sector',sector_options)

        if not selected_sector:
            merged_df3 = merged_df2.copy()
        else:
            merged_df3 = merged_df2[merged_df2['Sector'].isin(selected_sector)] 

        ##Filter for Share Unit range
        min_balance = st.sidebar.number_input(
        'Minimum Share Unit',
        min_value=int(merged_df3['Current Balance'].min()),
        max_value=int(merged_df3['Current Balance'].max()),
        value=int(merged_df3['Current Balance'].min())
        )
        max_balance = st.sidebar.number_input(
            'Maximum Share Unit',
            min_value=int(merged_df3['Current Balance'].min()),
            max_value=int(merged_df3['Current Balance'].max()),
            value=int(merged_df3['Current Balance'].max())
        )
            
        merged_df4 = merged_df3[(merged_df3['Current Balance'] >= min_balance) & (merged_df3['Current Balance'] <= max_balance)]

        ##Filter For Share value range
        min_balance = st.sidebar.number_input(
        'Minimum Share value',
        min_value=int(merged_df4['Value as of LTP'].min()),
        max_value=int(merged_df4['Value as of LTP'].max()),
        value=int(merged_df4['Value as of LTP'].min())
        )
        max_balance = st.sidebar.number_input(
            'Maximum Share value',
            min_value=int(merged_df4['Value as of LTP'].min()),
            max_value=int(merged_df4['Value as of LTP'].max()),
            value=int(merged_df4['Value as of LTP'].max())
        )
            
        merged_df5 = merged_df4[(merged_df4['Value as of LTP'] >= min_balance) & (merged_df4['Value as of LTP'] <= max_balance)]

        # Display the filtered dataset
        st.write(merged_df5)
        # scrip = merged_df['Scrip'].unique()
        # selected_scrip = st.sidebar.multiselect('Select Scrip',scrip)
        # if not selected_scrip:
        #     merged_df2 = merged_df.copy()
        # else:
        #     merged_df2 = merged_df2[merged_df2['Scrip'].isin(selected_scrip)]

       

        ###Porfolio Metrics
        st.header("Your Portfolio Key Metrics📊")
        col1,col2,col3,col4 = st.columns(4)

        total_scrip = merged_df5['Scrip'].nunique()
        total_unit = merged_df5['Current Balance'].sum()
        portfolio_value = merged_df5['Value as of LTP'].sum()
        no_of_sector_invested = merged_df5['Sector'].nunique()

        with col1:
           st.markdown(f"""
              <style>
                .card {{
                    text-align: left;
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    color: #4CAF50;
                    transition: all 0.3s ease;
                }}
                .card:hover {{
                    background-color: #ffffff;
                    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
                    color: #4CAF50;
                }}
            </style>
            <div class="card">
                <h3 style="font-size: 24px;">Total Companies</h3>
                <p style="font-size: 36px; color: #4CAF50; text-align:center;">{total_scrip:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
              <style>
                .card {{
                    text-align: left;
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    color: #4CAF50;
                    transition: all 0.3s ease;
                }}
                .card:hover {{
                    background-color: #ffffff;
                    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
                    color: #4CAF50;
                }}
            </style>
            <div class="card">
                <h3 style="font-size: 24px;">Total Units</h3>
                <p style="font-size: 36px; color: #2196F3; text-align:center;">{total_unit:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
              <style>
                .card {{
                    text-align: left;
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    color: #4CAF50;
                    transition: all 0.3s ease;
                }}
                .card:hover {{
                    background-color: #ffffff;
                    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
                    color: #4CAF50;
                }}
            </style>
            <div class="card">
                <h3 style="font-size: 24px;">Portfolio Value</h3>
                <p style="font-size: 36px; color: #FF5722; text-align:center;">RS.{portfolio_value:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
              <style>
                .card {{
                    text-align: left;
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    color: #4CAF50;
                    transition: all 0.3s ease;
                }}
                .card:hover {{
                    background-color: #ffffff;
                    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
                    color: #4CAF50;
                }}
            </style>
            <div class="card">
                <h3 style="font-size: 24px;">Sector Invested</h3>
                <p style="font-size: 36px; color: #386641; text-align:center;">{no_of_sector_invested:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.header("Share Unit Analysis📊")
        def format_label(value):
            """Format the label with 'K' for thousands and keep below 1000 as is."""
            if value >= 1000000:
                return f'{value / 1000000:.0f}M'
            elif value >= 1000:
                return f'{value / 1000:.0f}K'
            else:
                return f'{value:.0f}'
        col1,col2 =st.columns(2)
        with col1:
            st.subheader("High Unit Share")
            high_unit_share = merged_df5.nlargest(5, 'Current Balance')
            scrip_name = high_unit_share['Scrip'].values[0]
            unit_count = high_unit_share['Current Balance'].values[0]
            company_name = high_unit_share['Company'].values[0]
            sector_name = high_unit_share['Sector'].values[0]

            fig_high = px.bar(
                high_unit_share,
                x='Scrip',
                y='Current Balance',
                title="Top 10 High Unit Shares",
                labels={'Current Balance': 'Units'},
                color='Scrip',
                text_auto=True
            )
            st.plotly_chart(fig_high)
            st.write(f"The highest unit of share in your portfolio is {scrip_name} ({company_name}) with {unit_count} units, belonging to the {sector_name} sector.")


        with col2:
            st.subheader("Low Unit Share")
            low_unit_share = merged_df5.nsmallest(5, 'Current Balance')  # Top 5 scrips with the lowest unit shares

            scrip_name_low = low_unit_share['Scrip'].values[0]
            company_name_low = low_unit_share['Company'].values[0]
            unit_count_low = low_unit_share['Current Balance'].values[0]
            sector_name_low = low_unit_share['Sector'].values[0]

            fig_low = px.bar(
                low_unit_share,
                x='Scrip',
                y='Current Balance',
                title="Top 10 Low Unit Shares",
                labels={'Current Balance': 'Units'},
                color='Scrip',
                text_auto=True
            )
            st.plotly_chart(fig_low)
            st.write(f"The lowest unit of share in your portfolio is {scrip_name_low} ({company_name_low}) with {unit_count_low} units, belonging to the {sector_name_low} sector.")

        st.header("Share Value Analysis📊")
        col1,col2 =st.columns(2)
        with col1:
            st.subheader("High Value Share")
            
            # Get the top 10 scrips with the highest value as of LTP
            high_value_share = merged_df5.nlargest(5, 'Value as of LTP')
            
            # Extract the top scrip details
            scrip_name = high_value_share['Scrip'].values[0]
            value_amount = high_value_share['Value as of LTP'].values[0]
            company_name = high_value_share['Company'].values[0]
            sector_name = high_value_share['Sector'].values[0]
            
            # Create the bar chart for the highest value shares
            fig_high = px.bar(
                high_value_share,
                x='Scrip',
                y='Value as of LTP',
                title="Top 10 High Value Shares",
                labels={'Value as of LTP': 'Value'},
                color='Scrip',
                text_auto=True
            )
            fig_high.update_traces(
                text=[format_label(val) for val in high_value_share['Value as of LTP']],
                textposition='outside',     # Position the text outside the bars
                textfont=dict(size=11,color='black')      # Increase the font size
            )
            st.plotly_chart(fig_high)
            
            # Write the message with the top scrip details
            st.write(f"The highest value share in your portfolio is {scrip_name} ({company_name}) with a value of {value_amount}, belonging to the {sector_name} sector.")

        with col2:
            st.subheader("Low Value Share")
            
            # Get the top 10 scrips with the lowest value as of LTP
            low_value_share = merged_df5.nsmallest(5, 'Value as of LTP')
            
            # Extract the lowest scrip details
            scrip_name_low = low_value_share['Scrip'].values[0]
            value_amount_low = low_value_share['Value as of LTP'].values[0]
            company_name_low = low_value_share['Company'].values[0]
            sector_name_low = low_value_share['Sector'].values[0]
            
            # Create the bar chart for the lowest value shares
            fig_low = px.bar(
                low_value_share,
                x='Scrip',
                y='Value as of LTP',
                title="Top 10 Low Value Shares",
                labels={'Value as of LTP': 'Value'},
                color='Scrip',
                text_auto=True
            )
            fig_low.update_traces(
            text=[format_label(val) for val in low_value_share['Value as of LTP']],
            textposition='outside',
            textfont=dict(size=11,color='black')  )
            st.plotly_chart(fig_low)
            
            # Write the message with the lowest scrip details
            st.write(f"The lowest value share in your portfolio is {scrip_name_low} ({company_name_low}) with a value of {value_amount_low}, belonging to the {sector_name_low} sector.")
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