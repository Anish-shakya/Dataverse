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
                title="Top 5 High Unit Shares",
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
                title="Top 5 Low Unit Shares",
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
                title="Top 5 High Value Shares",
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
                title="Top 5 Low Value Shares",
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

        st.header("Sector Analysis📊")
        sector_unit_distribution = merged_df5.groupby('Sector')['Current Balance'].sum()
        sector_value_distribution = merged_df5.groupby('Sector')['Value as of LTP'].sum()

        col1,col2 = st.columns(2)

        with col1:
            # Unit Distribution Sector-wise (Percentage)
            fig_unit_distribution = px.pie(
                sector_unit_distribution.reset_index(),
                names='Sector',
                values='Current Balance',
                title='Unit Distribution Sector-wise (Percentage)',
                labels={'Current Balance': 'Units'},
                hole=0.6
            )
            fig_unit_distribution.update_traces(textinfo='percent', textfont=dict(size=14))
            st.plotly_chart(fig_unit_distribution)
            high_unit_sector = sector_unit_distribution.nlargest(5).reset_index()
            st.write(f"The sector with the highest share units is {sector_unit_distribution.idxmax()} with a total of {sector_unit_distribution.max():,.2f} units, making up {sector_unit_distribution.max()/sector_unit_distribution.sum()*100:.1f}% of the portfolio.")
        
        with col2:
            # Value Distribution Sector-wise (Percentage)
            fig_value_distribution = px.pie(
                sector_value_distribution.reset_index(),
                names='Sector',
                values='Value as of LTP',
                title='Value Distribution Sector-wise (Percentage)',
                labels={'Value as of LTP': 'Value'},
                hole=0.6
            )
            fig_value_distribution.update_traces(textinfo='percent', textfont=dict(size=14))
            st.plotly_chart(fig_value_distribution)
            st.write(f"The sector with the highest portfolio value is {sector_value_distribution.idxmax()} with a value of Rs.{sector_value_distribution.max():,.1f}, representing {sector_value_distribution.max()/sector_value_distribution.sum()*100:.1f}% of the total portfolio value.")

         # Display the table in Streamlit
        st.subheader("Sector-wise Top Share Analysis📊")
        # Calculate the top share in each sector
        top_shares_by_sector = merged_df5.loc[merged_df5.groupby('Sector')['Value as of LTP'].idxmax()]

        # Select relevant columns for the table
        sector_table = top_shares_by_sector[['Sector', 'Scrip', 'Company', 'Current Balance', 'Value as of LTP']]

        # Format the 'Value as of LTP' and 'Current Balance' columns for readability
        sector_table['Value as of LTP'] = sector_table['Value as of LTP'].apply(lambda x: f"Rs.{x:,.0f}")
        sector_table['Current Balance'] = sector_table['Current Balance'].apply(lambda x: f"{x:,.0f}")
        
        sector_table.reset_index(drop=True, inplace=True)
        sector_table.index = sector_table.index + 1

       
        st.table(sector_table)

        # Add explanation
        st.write("This table shows the top share (highest value) in each sector of your portfolio. The sectors are listed along with the corresponding top share, the number of units, and the value as per the latest trading price (LTP).")
        
        #########################
        st.header("Current Market Capital of Your Invested Shares 📊")
         # Ensure 'market capitalization (Rs)' column is numeric and handle any conversion if needed
        merged_df5['Market Capitalization (Rs)'] = pd.to_numeric(merged_df5['Market Capitalization (Rs)'], errors='coerce')

        top_5_df = merged_df5.nlargest(5, 'Current Balance')
        # Sort the DataFrame by 'Market Capitalization (Rs)' in descending order
        top_5_df = top_5_df.sort_values(by='Market Capitalization (Rs)', ascending=False)
        # Custom function to format the labels in billions
        def format_billions(value):
            if value >= 1e9:
                return f"{value / 1e9:.0f}B"
            elif value >= 1e6:
                return f"{value / 1e6:.1f}M"
            else:
                return f"{value:.1f}"
        # Apply formatting to each value in the dataset
        top_5_df['Formatted Market Capitalization'] = top_5_df['Market Capitalization (Rs)'].apply(format_billions)
        # Plot market capitalizationds
        fig_market_cap = px.bar(
            top_5_df,
            x='Scrip',
            y='Market Capitalization (Rs)',
            title='Market Capitalization of Your Top 5 Portfolio Shares',
            labels={'Market Capitalization (Rs)': 'Market Capitalization (Rs)'},
            color='Company',
            text_auto=True
        )

        # Customize the appearance of the chart
        fig_market_cap.update_layout(
            xaxis_title='Scrip',
            yaxis_title='Market Capitalization (Rs)',
            xaxis_tickangle=-45  # Tilt x-axis labels for better readability
        )
        max_y = top_5_df['Market Capitalization (Rs)'].max()
        fig_market_cap.update_yaxes(range=[0, max_y * 1.20])
        
        # Update data label font size and format
        fig_market_cap.update_traces(
            textfont_size=14# Adjust the font size as needed
        )

        st.plotly_chart(fig_market_cap)
        # Explanation
        top_share = top_5_df.iloc[0]
        scrip_name = top_share['Scrip']
        market_cap = top_share['Market Capitalization (Rs)']
        formatted_market_cap = f"{market_cap / 1e9:.1f}B"

        st.write(f"The chart above shows the market capitalization of the top 5 shares in your portfolio based on the highest units available. "
                f"The largest market capitalization share is {scrip_name}, with a market cap of {formatted_market_cap}. "
                f"This indicates that these shares have a significant presence in the market, reflecting their strong financial performance. "
                f"The chart helps you understand the market value of the shares you hold, allowing you to assess their impact on your overall investment portfolio.")
        
        ##################### low market cap analysis
        # Get the bottom 5 shares with the lowest 'Current Balance'
        low_5_df = merged_df5.nsmallest(5, 'Current Balance')

        # Sort the DataFrame by 'Market Capitalization (Rs)' in ascending order (lowest to highest)
        low_5_df = low_5_df.sort_values(by='Market Capitalization (Rs)', ascending=True)

        # Custom function to format the labels in billions
        def format_billions(value):
            if value >= 1e9:
                return f"{value / 1e9:.1f}B"
            elif value >= 1e6:
                return f"{value / 1e6:.1f}M"
            else:
                return f"{value:.1f}"

        # Apply formatting to each value in the dataset
        low_5_df['Formatted Market Capitalization'] = low_5_df['Market Capitalization (Rs)'].apply(format_billions)

        # Plot market capitalizations
        fig_low_market_cap = px.bar(
            low_5_df,
            x='Scrip',
            y='Market Capitalization (Rs)',
            title='Market Capitalization of Your Lowest 5 Portfolio Shares',
            labels={'Market Capitalization (Rs)': 'Market Capitalization (Rs)'},
            color='Company',
            text_auto=True
        )

        # Customize the appearance of the chart
        fig_low_market_cap.update_layout(
            xaxis_title='Scrip',
            yaxis_title='Market Capitalization (Rs)',
            xaxis_tickangle=-45  # Tilt x-axis labels for better readability
        )

        # Update y-axis range
        max_y = low_5_df['Market Capitalization (Rs)'].max()
        fig_low_market_cap.update_yaxes(range=[0, max_y * 1.20])

        # Update data label font size and format
        fig_low_market_cap.update_traces(
            textfont_size=14,  # Adjust the font size as needed
        )

        # Show chart
        st.plotly_chart(fig_low_market_cap)

        # Explanation
        low_share = low_5_df.iloc[0]
        scrip_name = low_share['Scrip']
        market_cap = low_share['Market Capitalization (Rs)']
        formatted_market_cap = f"{market_cap / 1e9:.1f}B"

        st.write(f"The chart above displays the market capitalization of the 5 shares in your portfolio with the lowest units. "
                f"The share with the smallest market capitalization is {scrip_name}, with a market cap of {formatted_market_cap}. "
                f"This indicates that these shares have a relatively lower market value compared to others in your portfolio. "
                f"The chart helps you understand the market value of these lesser valued shares and assess their impact on your investment portfolio.")
        

        st.title('Share Volatility Analysis')
        col1,col2 = st.columns(2)
        with col1:
            # Sort the DataFrame by 'Price Change (%)' in descending order
            increase_change_share = merged_df5.nlargest(5, 'Price Change (%)')

            fig_high = px.bar(
                    increase_change_share,
                    x='Scrip',
                    y='Price Change (%)',
                    title="Increased Share",
                    labels={'Price Change (%)': 'Percent Increase'},
                    color='Scrip',
                    text_auto=True
                )
            st.plotly_chart(fig_high)
        with col2:
            # Sort the DataFrame by 'Price Change (%)' in descending order
            decrease_change_share = merged_df5.nsmallest(5, 'Price Change (%)')

            fig_high = px.bar(
                    decrease_change_share,
                    x='Scrip',
                    y='Price Change (%)',
                    title="Increased Share",
                    labels={'Price Change (%)': 'Percent Decreased'},
                    color='Scrip',
                    text_auto=True
                )
            st.plotly_chart(fig_high)
        # Explanation
        st.write(
            "The chart above shows the volatility of shares in your portfolio based on the Price Change Percentage. "
            "Shares with higher Price Change (%) values indicate greater volatility, meaning their prices are fluctuating more significantly. "
            "You should monitor these volatile shares closely as they may represent higher risk or opportunity in the market."
        )




        st.header("Correlation Analysis")
        # Ensure columns are numeric
        merged_df5['Market Capitalization (Rs)'] = pd.to_numeric(merged_df5['Market Capitalization (Rs)'], errors='coerce')
        merged_df5['Listed Share'] = pd.to_numeric(merged_df5['Listed Share'], errors='coerce')

        col1, col2 = st.columns(2)

        with col1:
            # Calculate the correlation matrix for LTP and Market Capitalization
            corr_matrix_ltp_market_cap = merged_df5[['Last Transaction Price (LTP)', 'Market Capitalization (Rs)']].corr()
            correlation_ltp_market_cap = corr_matrix_ltp_market_cap.iloc[0, 1]
            
            # Set up the matplotlib figure
            plt.figure(figsize=(8, 6))
            
            # Create the heatmap
            sns.heatmap(corr_matrix_ltp_market_cap, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            
            # Set the title and display the heatmap
            plt.title('Correlation Heatmap: LTP vs Market Capitalization')
            st.pyplot(plt.gcf())
            
            # Display correlation coefficient
            st.write(f"The correlation coefficient between Last Transaction Price (LTP) and Market Capitalization is: {correlation_ltp_market_cap:.2f}")


        with col2:
            # Calculate the correlation matrix for LTP and Listed Share
            corr_matrix_ltp_listed_share = merged_df5[['Last Transaction Price (LTP)', 'Listed Share']].corr()
            correlation_ltp_listed_share = corr_matrix_ltp_listed_share.iloc[0, 1]
            
            # Set up the matplotlib figure
            plt.figure(figsize=(8, 6))
            
            # Create the heatmap
            sns.heatmap(corr_matrix_ltp_listed_share, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            
            # Set the title and display the heatmap
            plt.title('Correlation Heatmap: LTP vs Listed Share')
            st.pyplot(plt.gcf())
            
            # Display correlation coefficient
            st.write(f"The correlation coefficient between Last Transaction Price (LTP) and Listed Shares is: {correlation_ltp_listed_share:.2f}")

     
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