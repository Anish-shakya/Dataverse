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


class PharmaceuticalSalesAnalysis:
    expected_columns = {
        "ID": "object",
        "Date": "datetime64[ns]",
        "Product_Name": "object",
        "Category": "object",
        "Payment_Mode": "object",
        "Quantity": "int64",
        "Cost_Price": "float64",
        "Selling_Price": "float64",
        "Amount": "float64",
        "Discount": "float64",
        "Net_Amount": "float64"
    }

    def __init__(self, dataframe):
        self.df = dataframe
        self.convert_dtypes()

    def convert_dtypes(self):
        """Convert columns to the expected data types."""
        for col, dtype in PharmaceuticalSalesAnalysis.expected_columns.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype, errors='ignore')

    def validate_schema(self):
        """Validate if the dataframe has the required columns with the correct data types."""
        missing_columns = [col for col in PharmaceuticalSalesAnalysis.expected_columns if col not in self.df.columns]
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        for col, dtype in PharmaceuticalSalesAnalysis.expected_columns.items():
            if self.df[col].dtype != dtype:
                return False, f"Column '{col}' does not match the expected data type '{dtype}'."
        
        return True, None
    
    def preprocess(self):
        """Preprocess the data (you can add more logic here as needed)."""
        # Additional preprocessing steps can be added here

        ## Calculating Profit
        self.df['Profit'] = ((self.df['Selling_Price']-self.df['Cost_Price'])*self.df['Quantity'])

        ## Calculating Margin
        self.df['Margin'] = self.df['Profit'] / self.df['Selling_Price']

        ## Date column 
        self.df['Date'] = pd.to_datetime(self.df['Date'],errors='coerce')

        ## Year , Month and Days of Week
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['MonthName'] = self.df['Date'].dt.month_name()
        self.df['DayName'] = self.df['Date'].dt.day_name()

        return True
    
    def perform_eda(self):
        """Perform exploratory data analysis (EDA) on the dataset."""
        df = self.df

        col1,col2 = st.columns((2))

        ### getting MIN and MAX date from the date
        startDate = pd.to_datetime(df['Date']).min()
        endDate = pd.to_datetime(df['Date']).max()

        with col1:
            date1 = pd.to_datetime(st.date_input("Start Date", startDate))
        
        with col2:
            date2 = pd.to_datetime(st.date_input("End Date", endDate))
        
        date1 = pd.to_datetime(date1)
        date2 = pd.to_datetime(date2)

        df = df[(df['Date'] >= date1) & (df['Date'] <=date2)].copy()
        

        ### side bar filter pane
        st.sidebar.header("Choose Your Filters:")
        
        ## Create for Category
        categories = df['Category'].unique()
        selected_categories = st.sidebar.multiselect('Select Categories',categories)
        if not selected_categories:
            df2 =df.copy()
        else:
            df2 = df[df['Category'].isin(selected_categories)]

        ## Create For Payment Mode
        paymentmode = df['Payment_Mode'].unique()
        selected_paymentmode=st.sidebar.multiselect('Select Payment Mode',paymentmode)
        if not selected_paymentmode:
            df3 = df2.copy()
        else:
            df3 = df2[df['Payment_Mode'].isin(selected_paymentmode)]
        
        ## Month Name
        month = df['MonthName'].unique()
        selected_month=st.sidebar.multiselect('Select Month',month)
        if not selected_month:
            df4 = df3.copy()
        else:
            df4 = df3[df['MonthName'].isin(selected_month)]

        ## Day Name
        dayname = df['DayName'].unique()
        selected_dayname=st.sidebar.multiselect('Select Day Of Week',dayname)
        if not selected_dayname:
            df5 = df4.copy()
        else:
            df5 = df4[df['DayName'].isin(selected_dayname)]

        st.write(df5)
        st.header("Sales Metrics📈")

        total_sales = df5['Net_Amount'].sum()
        total_profit = df5['Profit'].sum()
        total_discount = df5['Discount'].sum()

        # Display KPIs in three columns
        col1, col2, col3 = st.columns(3)

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
                <h3 style="font-size: 24px;">Total Sales</h3>
                <p style="font-size: 36px; color: #4CAF50;">Rs.{total_sales:,.2f}</p>
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
                <h3 style="font-size: 24px;">Total Profit </h3>
                <p style="font-size: 36px; color: #2196F3;">Rs.{total_profit:,.2f}</p>
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
                        
            <div class ="card">
                <h3 style="font-size: 24px;">Total Discount</h3>
                <p style="font-size: 36px; color: #FF5722;">Rs.{total_discount:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        st.header("Sales Key Insights📊")
              
        col4,col5 = st.columns(2)

        with col4:
              #Sales by Month
            sales_by_month = df5.groupby(by=['MonthName'],as_index=False)['Net_Amount'].sum()
            sales_by_month = sales_by_month.sort_values('MonthName', key=lambda x: pd.to_datetime(x, format='%B'))
            st.subheader("Net Sales Over Month")
            fig = px.bar(sales_by_month,x='MonthName',y='Net_Amount',
                         text=['Rs{:,.2f}'.format(x) for x in sales_by_month['Net_Amount']]
                         )
            st.plotly_chart(fig,use_container_width=True,height=200)
        
        with col5:
            #Sales by dayname
            sales_by_dayname= df5.groupby(by=['DayName'],as_index=False)['Net_Amount'].sum()
            st.subheader("Net Sales Over Day of Week")
            fig = px.bar(sales_by_dayname,x='DayName',y='Net_Amount',
                         text=['Rs{:,.2f}'.format(x) for x in sales_by_dayname['Net_Amount']],
                         )
            st.plotly_chart(fig,use_container_width=True,height=200)


        col6,col7= st.columns(2)
        with col6:
            ## Payment Mode Sale distribution
            sales_by_paymentmode = df5.groupby(by=['Payment_Mode'],as_index=False)['Net_Amount'].sum()
            st.subheader("Payment Mode Sales Distribution")
            fig = px.pie(sales_by_paymentmode,values='Net_Amount', names='Payment_Mode',hole=0.6)
            fig.update_traces(text=df5['Payment_Mode'].unique(),textposition="outside")
            st.plotly_chart(fig)
        
        with col7:
            ## Top 5 Sales by Category
            # Aggregate sales by category
            sales_by_category = df5.groupby(by='Category', as_index=False)['Net_Amount'].sum()
            # Sort and select the top 5 categories
            top_categories = sales_by_category.sort_values(by='Net_Amount', ascending=False).head(5)
            st.subheader("Top 5 Sales by Category")
            fig = px.pie(top_categories, values='Net_Amount', names='Category', hole=0.6)
            fig.update_traces(textinfo='label+percent', textposition='outside')  # Ensure correct display of labels
            st.plotly_chart(fig)


        sales_trend = df5.groupby(by='Date', as_index=False)['Net_Amount'].sum()
    
        # Calculate highest and lowest sales
        highest_sales_date = sales_trend.loc[sales_trend['Net_Amount'].idxmax()]
        lowest_sales_date = sales_trend.loc[sales_trend['Net_Amount'].idxmin()]
        average_sales = sales_trend['Net_Amount'].mean()

        st.subheader("Sales Trend Over Time")

        # Create the line chart
        fig = px.line(sales_trend, x='Date', y='Net_Amount', markers=True, template='plotly_dark')

        # Add markers for highest and lowest sales
        fig.add_trace(go.Scatter(
            x=[highest_sales_date['Date']],
            y=[highest_sales_date['Net_Amount']],
            mode='markers+text',
            name='Highest Sales',
            text=['Highest Sales: Rs{:,.2f}'.format(highest_sales_date['Net_Amount'])],
            textposition='top center',
            marker=dict(color='red', size=10)
        ))

        fig.add_trace(go.Scatter(
            x=[lowest_sales_date['Date']],
            y=[lowest_sales_date['Net_Amount']],
            mode='markers+text',
            name='Lowest Sales',
            text=['Lowest Sales: Rs{:,.2f}'.format(lowest_sales_date['Net_Amount'])],
            textposition='bottom center',
            marker=dict(color='blue', size=10)
        ))

        fig.add_trace(go.Scatter(
        x=sales_trend['Date'],
        y=[average_sales] * len(sales_trend),
        mode='lines',
        name='Average Sales',
        line=dict(color='green', dash='dash')
        ))

        
        # Update layout with y-axis range adjustment
        fig.update_layout(
            yaxis=dict(
                title='Net Sales',
                range=[sales_trend['Net_Amount'].min() - 0.3 * sales_trend['Net_Amount'].max(), 
                    sales_trend['Net_Amount'].max() + 0.5 * sales_trend['Net_Amount'].max()]
            ),
            xaxis_title='Date',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True, height=600)  # Adjust the height as needed

        ######to 10 sold products
        top_10_soldproducts = df5.groupby('Product_Name', as_index=False)['Quantity'].sum()
        top_10_soldproducts = top_10_soldproducts.sort_values('Quantity', ascending=False).head(10)

            # Plot
        st.subheader("Top 10 Most Sold Products")
        fig = px.bar(top_10_soldproducts, x='Product_Name', y='Quantity',
                        text=['{:,.0f}'.format(x) for x in top_10_soldproducts['Quantity']])
        st.plotly_chart(fig, use_container_width=True)

        st.header("Profit Key Insights📊")
        col8,col9 = st.columns(2)

        with col8:
           #Profit by Month
            profit_by_month = df5.groupby(by=['MonthName'],as_index=False)['Profit'].sum()
            profit_by_month = profit_by_month.sort_values('MonthName', key=lambda x: pd.to_datetime(x, format='%B'))
            st.subheader("Net Profit Over Month")
            fig = px.bar(profit_by_month,x='MonthName',y='Profit',
                         text=['Rs{:,.2f}'.format(x) for x in profit_by_month['Profit']]
                         )
            st.plotly_chart(fig,use_container_width=True,height=200)
        with col9:
            ## Top 5 profit by Category
            # Aggregate profit by category
            profit_by_category = df5.groupby(by='Category', as_index=False)['Profit'].sum()
            # Sort and select the top 5 categories
            top_categories = profit_by_category.sort_values(by='Profit', ascending=False).head(5)
            st.subheader("Top 5 Profitable Category")
            fig = px.pie(top_categories, values='Profit', names='Category', hole=0.6)
            fig.update_traces(textinfo='label+percent', textposition='outside')  # Ensure correct display of labels
            st.plotly_chart(fig)
        col10,col11=st.columns(2)
        with col10:
            top_5_products = df5.groupby('Product_Name', as_index=False)['Profit'].sum()
            top_5_products = top_5_products.sort_values('Profit', ascending=False).head(5)

                # Plot
            st.subheader("Most Profitable Products")
            fig = px.bar(top_5_products, x='Product_Name', y='Profit',
                            text=['Rs{:,.2f}'.format(x) for x in top_5_products['Profit']])
            st.plotly_chart(fig, use_container_width=True)

        with col11:
            top_5_products = df5.groupby('Product_Name', as_index=False)['Profit'].sum()
            top_5_products = top_5_products.sort_values('Profit', ascending=True).head(5)

                # Plot
            st.subheader("Least Profitable Products")
            fig = px.bar(top_5_products, x='Product_Name', y='Profit',
                            text=['Rs{:,.2f}'.format(x) for x in top_5_products['Profit']])
            st.plotly_chart(fig, use_container_width=True)

        # Grouping data by Date and summing the Profit
        profit_trend = df5.groupby(by='Date', as_index=False)['Profit'].sum()

        # Calculate highest and lowest profit days
        highest_profit_date = profit_trend.loc[profit_trend['Profit'].idxmax()]
        lowest_profit_date = profit_trend.loc[profit_trend['Profit'].idxmin()]
        average_profit = profit_trend['Profit'].mean()

        st.subheader("Profit Trend Over Time")

        # Create the line chart for profit trend
        fig = px.line(profit_trend, x='Date', y='Profit', markers=True, template='plotly_dark')

        # Add markers for highest and lowest profit
        fig.add_trace(go.Scatter(
            x=[highest_profit_date['Date']],
            y=[highest_profit_date['Profit']],
            mode='markers+text',
            name='Highest Profit',
            text=['Highest Profit: Rs{:,.2f}'.format(highest_profit_date['Profit'])],
            textposition='top center',
            marker=dict(color='red', size=10)
        ))

        fig.add_trace(go.Scatter(
            x=[lowest_profit_date['Date']],
            y=[lowest_profit_date['Profit']],
            mode='markers+text',
            name='Lowest Profit',
            text=['Lowest Profit: Rs{:,.2f}'.format(lowest_profit_date['Profit'])],
            textposition='bottom center',
            marker=dict(color='blue', size=10)
        ))


        # Update layout with y-axis range adjustment
        fig.update_layout(
            yaxis=dict(
                title='Profit',
                range=[profit_trend['Profit'].min() - 0.3 * profit_trend['Profit'].max(),
                    profit_trend['Profit'].max() + 0.5 * profit_trend['Profit'].max()]
            ),
            xaxis_title='Date',
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True, height=600)  # Adjust the height as needed

        st.header('Correlation Coefficient Analysis 📊')
        
        col12,col13 = st.columns(2)
        # Calculate the correlation matrix
        with col12:
            # Create and display heatmap
            correlation1 = df5[['Net_Amount', 'Profit']].corr().iloc[0, 1]
            corr_matrix = df5[['Net_Amount', 'Profit']].corr()
            st.subheader('Correlation Sales And Profit')
            st.write(f"The correlation coefficient between Net Sales and Profit is: {correlation1:.2f}")
            
            # Set up the matplotlib figure
            plt.figure(figsize=(8, 6))
            
            # Create the heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            
            # Set the title and display the heatmap
            plt.title('Correlation Heatmap')
            st.pyplot(plt.gcf())

        with col13:
             # Calculate correlation coefficient
            correlation2 = df5[['Discount', 'Profit']].corr().iloc[0, 1]
            

            # Calculate the correlation matrix
            corr_matrix = df5[['Discount', 'Profit']].corr()

            # Create and display heatmap
            st.subheader('Correlation Discount And Profit')
            st.write(f"The correlation coefficient between Discount and Profit is: {correlation2:.2f}")
            
            # Set up the matplotlib figure
            plt.figure(figsize=(8, 6))
            
            # Create the heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
            
            # Set the title and display the heatmap
            plt.title('Correlation Heatmap')
            st.pyplot(plt.gcf())

        #col14,col15 =st.columns(2)       
            
    
    def show(self):
        """Display the analysis after validation."""
        valid, message = self.validate_schema()
        if valid:
            ## Check if Data is pre Processed or not
            if PharmaceuticalSalesAnalysis.preprocess(self):
                self.perform_eda()
            # Additional analysis logic can be added here
        else:
            st.sidebar.error(f"Schema does not match. {message}")
            data = [
            {"Field": "ID", "Description": "Sale ID."},
            {"Field": "Date", "Description": "The date of the sale."},
            {"Field": "Product_Name", "Description": "The name of the product sold."},
            {"Field": "Category", "Description": "The category of the product sold."},
            {"Field": "Payment_Mode", "Description": "Mode of payment made by customer."},
            {"Field": "Quantity", "Description": "The number of units sold."},
            {"Field": "Cost_Price", "Description": "The cost price per unit of the product."},
            {"Field": "Selling_Price", "Description": "The selling price per unit of the product."},
            {"Field": "Amount", "Description": "The amount of product before discount."},
            {"Field": "Discount", "Description": "The discount applied to the product, if any."},
            {"Field": "Net_Amount", "Description": "The net amount after discount."},
            ]

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(data)
            st.write("Your Dataset must contain following columns with strict naming convention in order to proceed")
            # Display the DataFrame as a table in Streamlit
            st.table(df.style.hide(axis="index"))

