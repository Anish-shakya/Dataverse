import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class SalesAnalysis:
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
        for col, dtype in SalesAnalysis.expected_columns.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype, errors='ignore')

    def validate_schema(self):
        """Validate if the dataframe has the required columns with the correct data types."""
        missing_columns = [col for col in SalesAnalysis.expected_columns if col not in self.df.columns]
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        for col, dtype in SalesAnalysis.expected_columns.items():
            if self.df[col].dtype != dtype:
                return False, f"Column '{col}' does not match the expected data type '{dtype}'."
        
        return True, None

    def preprocess(self):
        """Preprocess the data (you can add more logic here as needed)."""
        self.calculate_profit_margin()
        self.process_dates()

    def calculate_profit_margin(self):
        """Calculate profit and margin."""
        self.df['Profit'] = (self.df['Selling_Price'] - self.df['Cost_Price']) * self.df['Quantity']
        self.df['Margin'] = self.df['Profit'] / self.df['Selling_Price']

    def process_dates(self):
        """Convert date column and extract additional date features."""
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['MonthName'] = self.df['Date'].dt.month_name()
        self.df['DayName'] = self.df['Date'].dt.day_name()

    def perform_eda(self):
        """Perform exploratory data analysis (EDA) on the dataset."""
        col1, col2 = st.columns(2)

        startDate = pd.to_datetime(self.df['Date']).min()
        endDate = pd.to_datetime(self.df['Date']).max()

        with col1:
            date1 = pd.to_datetime(st.date_input("Start Date", startDate))
        
        with col2:
            date2 = pd.to_datetime(st.date_input("End Date", endDate))
        
        filtered_df = self.filter_data_by_date(date1, date2)

        st.sidebar.header("Choose Your Filters:")
        filtered_df = self.apply_filters(filtered_df)

        self.display_kpis(filtered_df)
        self.display_sales_insights(filtered_df)
        self.display_profit_insights(filtered_df)
        self.display_profit_trend(filtered_df)
        self.display_top_profit_products(filtered_df)


    def filter_data_by_date(self, start_date, end_date):
        """Filter data based on selected date range."""
        return self.df[(self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)].copy()

    def apply_filters(self, df):
        """Apply sidebar filters for categories and payment modes."""
        categories = df['Category'].unique()
        selected_categories = st.sidebar.multiselect('Select Categories', categories)
        df = df[df['Category'].isin(selected_categories)] if selected_categories else df

        payment_modes = df['Payment_Mode'].unique()
        selected_payment_modes = st.sidebar.multiselect('Select Payment Mode', payment_modes)
        df = df[df['Payment_Mode'].isin(selected_payment_modes)] if selected_payment_modes else df

        month_names = df['MonthName'].unique()
        selected_months = st.sidebar.multiselect('Select Month', month_names)
        df = df[df['MonthName'].isin(selected_months)] if selected_months else df

        day_names = df['DayName'].unique()
        selected_days = st.sidebar.multiselect('Select Day Of Week', day_names)
        df = df[df['DayName'].isin(selected_days)] if selected_days else df

        return df

    def display_kpis(self, df):
        """Display key performance indicators."""
        total_sales = df['Net_Amount'].sum()
        total_profit = df['Profit'].sum()
        total_discount = df['Discount'].sum()

        col1, col2, col3 = st.columns(3)
        self.display_kpi_card(col1, "Total Sales", total_sales)
        self.display_kpi_card(col2, "Total Profit", total_profit, color="#2196F3")
        self.display_kpi_card(col3, "Total Discount", total_discount, color="#FF5722")

    def display_kpi_card(self, col, title, value, color="#4CAF50"):
        """Display an individual KPI card."""
        col.markdown(f"""
            <style>
            .card {{
                text-align: left;
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                color: {color};
                transition: all 0.3s ease;
            }}
            .card:hover {{
                background-color: #ffffff;
                box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
                color: {color};
            }}
            </style>
            <div class="card">
                <h3 style="font-size: 24px;">{title}</h3>
                <p style="font-size: 36px; color: {color};">Rs.{value:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    def display_sales_insights(self, df):
        """Display sales insights through various visualizations."""
        st.header("Sales Key Insights📊")
        col1, col2 = st.columns(2)
        col3,col4 = st.columns(2)
        with col1:
            sales_by_month = df.groupby(by=['MonthName'], as_index=False)['Net_Amount'].sum().sort_values('MonthName', key=lambda x: pd.to_datetime(x, format='%B'))
            self.plot_bar_chart(col1, sales_by_month, "Net Sales Over Month", 'MonthName', 'Net_Amount')
             # Insight for sales by month
            max_sales_month = sales_by_month.loc[sales_by_month['Net_Amount'].idxmax()]['MonthName']
            max_sales_value = sales_by_month['Net_Amount'].max()
            st.write(f"**Insight:** The highest sales occurred in **{max_sales_month}**, with a total of **{max_sales_value:,.2f}**.")
            
        with col2:
            sales_by_dayname = df.groupby(by=['DayName'], as_index=False)['Net_Amount'].sum()
            self.plot_bar_chart(col2, sales_by_dayname, "Net Sales Over Day of Week", 'DayName', 'Net_Amount')
            # Insight for sales by day of the week
            max_sales_day = sales_by_dayname.loc[sales_by_dayname['Net_Amount'].idxmax()]['DayName']
            max_sales_value_day = sales_by_dayname['Net_Amount'].max()
            st.write(f"**Insight:** The highest sales happened on **{max_sales_day}**, totaling **{max_sales_value_day:,.2f}**.")

        with col3:
            sales_by_paymentmode = df.groupby(by=['Payment_Mode'], as_index=False)['Net_Amount'].sum()
            self.plot_pie_chart(col1, sales_by_paymentmode, "Payment Mode Sales Distribution", 'Payment_Mode', 'Net_Amount')
            # Insight for sales by payment mode
            max_sales_paymentmode = sales_by_paymentmode.loc[sales_by_paymentmode['Net_Amount'].idxmax()]['Payment_Mode']
            max_sales_value_paymentmode = sales_by_paymentmode['Net_Amount'].max()
            st.write(f"**Insight:** Most sales were made through **{max_sales_paymentmode}**, contributing **{max_sales_value_paymentmode:,.2f}** in total sales.")
        with col4:
            top_categories = df.groupby('Category', as_index=False)['Net_Amount'].sum().sort_values(by='Net_Amount', ascending=False).head(5)
            self.plot_pie_chart(col2, top_categories, "Top 5 Sales by Category", 'Category', 'Net_Amount')
            # Insight for top categories
            top_category = top_categories.iloc[0]['Category']
            top_category_sales = top_categories.iloc[0]['Net_Amount']
            st.write(f"**Insight:** The category with the highest sales is **{top_category}**, with a total of **{top_category_sales:,.2f}** in sales.")

        self.display_sales_trend(df)
        self.display_top_sold_products(df)

    def plot_bar_chart(self, col, data, title, x, y):
        """Plot a bar chart."""
        col.subheader(title)
        fig = px.bar(data, x=x, y=y, text=[f'Rs{amount:,.2f}' for amount in data[y]])
        col.plotly_chart(fig, use_container_width=True)

    def plot_pie_chart(self, col, data, title, names, values):
        """Plot a pie chart."""
        col.subheader(title)
        fig = px.pie(data, names=names, values=values, hole=0.6)
        fig.update_traces(textinfo='label+percent', textposition='outside')
        col.plotly_chart(fig)

    def display_sales_trend(self, df):
        """Display sales trend over time."""
        st.subheader("Sales Trend Over Time")
        sales_trend = df.groupby(by='Date', as_index=False)['Net_Amount'].sum()

        highest_sales_date = sales_trend.loc[sales_trend['Net_Amount'].idxmax()]
        lowest_sales_date = sales_trend.loc[sales_trend['Net_Amount'].idxmin()]
        average_sales = sales_trend['Net_Amount'].mean()

        fig = px.line(sales_trend, x='Date', y='Net_Amount', markers=True, template='plotly_dark')
        fig.add_trace(go.Scatter(x=[highest_sales_date['Date']], y=[highest_sales_date['Net_Amount']],
                                  mode='markers+text', name='Highest Sales', text=[f'Highest Sales: Rs{highest_sales_date["Net_Amount"]:,.2f}'],
                                  textposition='top center', marker=dict(color='red', size=10)))
        fig.add_trace(go.Scatter(x=[lowest_sales_date['Date']], y=[lowest_sales_date['Net_Amount']],
                                  mode='markers+text', name='Lowest Sales', text=[f'Lowest Sales: Rs{lowest_sales_date["Net_Amount"]:,.2f}'],
                                  textposition='bottom center', marker=dict(color='green', size=10)))

        st.plotly_chart(fig)
        # Shortened explanation
        st.write(f"""
        **Sales Trend Insight:**
        
        This chart displays the daily sales trend over time. The **highest sales** occurred on **{highest_sales_date['Date']}** with a total of 
        **Rs{highest_sales_date['Net_Amount']:,.2f}**, while the **lowest sales** were on **{lowest_sales_date['Date']}**, amounting to **Rs{lowest_sales_date['Net_Amount']:,.2f}**.
        The **average daily sales** during this period were **Rs{average_sales:,.2f}**. Monitoring these trends helps identify peak sales days and potential low points.
        """)

    def display_top_sold_products(self, df):
        """Display top sold products."""
        st.subheader("Top Sold Products")
        top_products = df.groupby('Product_Name', as_index=False)['Quantity'].sum().sort_values(by='Quantity', ascending=False).head(5)
        fig = px.bar(top_products, x='Product_Name', y='Quantity', text='Quantity')
        st.plotly_chart(fig)
        # Explanation for the chart
        st.write(f"""
        **Top Sold Products Insight:**
        
        This chart shows the top 5 products by quantity sold. The product with the highest sales is **{top_products.iloc[0]['Product_Name']}** 
        with a total of **{top_products.iloc[0]['Quantity']}** units sold. Understanding which products are most popular can help 
        focus inventory management and marketing efforts on these high-demand items.
        """)

    def display_profit_insights(self, df):
        """Display profit insights through various visualizations."""
        st.header("Profit Insights💰")
        col1,col2 = st.columns(2)
        with col1:
            profit_by_month = df.groupby(by=['MonthName'], as_index=False)['Profit'].sum().sort_values('MonthName', key=lambda x: pd.to_datetime(x, format='%B'))
            self.plot_bar_chart(st, profit_by_month, "Profit Over Month", 'MonthName', 'Profit')
            # Explanation for profit by month
            highest_profit_month = profit_by_month.loc[profit_by_month['Profit'].idxmax()]['MonthName']
            highest_profit_value = profit_by_month['Profit'].max()
            st.write(f"""
            **Monthly Profit Insight:**
            The highest profit was recorded in **{highest_profit_month}**, with a total profit of **Rs{highest_profit_value:,.2f}**. 
            This month likely had a boost in sales or higher-margin products driving profitability.
            """)
        with col2:
            profit_by_category = df.groupby(by='Category', as_index=False)['Profit'].sum().sort_values(by='Profit', ascending=False).head(5)
            self.plot_pie_chart(st, profit_by_category, "Profit Distribution by Category", 'Category', 'Profit')
            # Explanation for profit by category
            top_category = profit_by_category.iloc[0]['Category']
            top_category_profit = profit_by_category.iloc[0]['Profit']
            st.write(f"""
            **Category Profit Insight:**
            The top contributing category is **{top_category}**, generating a profit of **Rs{top_category_profit:,.2f}**. 
            Focusing on this category could further boost overall profitability.
            """)
    
    def display_top_profit_products(self, df):
        """Display top Profit products."""
        st.subheader("Top Profit Products")
        
        # Group by product and calculate sum of profit
        top_products = df.groupby('Product_Name', as_index=False)['Profit'].sum()
        
        # Round the Profit column to 2 decimal places
        top_products['Profit'] = top_products['Profit'].round(2)
        
        # Sort by Profit in descending order and take the top 5
        top_products = top_products.sort_values(by='Profit', ascending=False).head(5)
        
        # Create the bar chart
        fig = px.bar(top_products, x='Product_Name', y='Profit', text='Profit')
        
        # Plot the chart
        st.plotly_chart(fig)
        # Explanation of the chart
        st.write(f"""
        **Top Profit Products Insight:**
        
        This chart displays the top 5 products based on their total profit. The product with the highest profit is **{top_products.iloc[0]['Product_Name']}**, 
        generating a total profit of **Rs{top_products.iloc[0]['Profit']:,.2f}**. Understanding which products contribute most to profitability helps in 
        focusing sales efforts and inventory management on these key items.
        """)


    def display_profit_trend(self, df):
        """Display Profit trend over time."""
        st.subheader("Profit Trend Over Time")
        sales_trend = df.groupby(by='Date', as_index=False)['Profit'].sum()

        highest_profit_date = sales_trend.loc[sales_trend['Profit'].idxmax()]
        lowest_profit_date = sales_trend.loc[sales_trend['Profit'].idxmin()]
        average_profit = sales_trend['Profit'].mean()

        fig = px.line(sales_trend, x='Date', y='Profit', markers=True, template='plotly_dark')
        fig.add_trace(go.Scatter(x=[highest_profit_date['Date']], y=[highest_profit_date['Profit']],
                                  mode='markers+text', name='Highest Profit', text=[f'Highest Profit: Rs{highest_profit_date["Profit"]:,.2f}'],
                                  textposition='top center', marker=dict(color='red', size=10)))
        fig.add_trace(go.Scatter(x=[lowest_profit_date['Date']], y=[lowest_profit_date['Profit']],
                                  mode='markers+text', name='Lowest Profit', text=[f'Lowest Profit: Rs{lowest_profit_date["Profit"]:,.2f}'],
                                  textposition='bottom center', marker=dict(color='green', size=10)))

        st.plotly_chart(fig)
        # Explanation for the chart
        st.write(f"""
        **Profit Trend Insight:**

        This chart shows the fluctuation of profits over time. The **highest profit** was recorded on **{highest_profit_date['Date']}** 
        with a total of **Rs{highest_profit_date['Profit']:,.2f}**, while the **lowest profit** occurred on **{lowest_profit_date['Date']}** 
        with a profit of **Rs{lowest_profit_date['Profit']:,.2f}**. The **average daily profit** over the period is **Rs{average_profit:,.2f}**.
        Monitoring these trends helps identify profitable periods and potential areas for improvement.
            """)


    def show(self):
        """Show the analysis."""
        is_valid, error_message = self.validate_schema()
        if not is_valid:
            st.error(error_message)
            return

        self.preprocess()
        self.perform_eda()
