import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


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

        # self.display_correlation_analysis(filtered_df)

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

        sales_by_month = df.groupby(by=['MonthName'], as_index=False)['Net_Amount'].sum().sort_values('MonthName', key=lambda x: pd.to_datetime(x, format='%B'))
        self.plot_bar_chart(col1, sales_by_month, "Net Sales Over Month", 'MonthName', 'Net_Amount')

        sales_by_dayname = df.groupby(by=['DayName'], as_index=False)['Net_Amount'].sum()
        self.plot_bar_chart(col2, sales_by_dayname, "Net Sales Over Day of Week", 'DayName', 'Net_Amount')

        sales_by_paymentmode = df.groupby(by=['Payment_Mode'], as_index=False)['Net_Amount'].sum()
        self.plot_pie_chart(col1, sales_by_paymentmode, "Payment Mode Sales Distribution", 'Payment_Mode', 'Net_Amount')

        top_categories = df.groupby('Category', as_index=False)['Net_Amount'].sum().sort_values(by='Net_Amount', ascending=False).head(5)
        self.plot_pie_chart(col2, top_categories, "Top 5 Sales by Category", 'Category', 'Net_Amount')

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

    def display_top_sold_products(self, df):
        """Display top sold products."""
        st.subheader("Top Sold Products")
        top_products = df.groupby('Product_Name', as_index=False)['Quantity'].sum().sort_values(by='Quantity', ascending=False).head(5)
        fig = px.bar(top_products, x='Product_Name', y='Quantity', text='Quantity', title="Top Sold Products")
        st.plotly_chart(fig)

    def display_profit_insights(self, df):
        """Display profit insights through various visualizations."""
        st.header("Profit Insights💰")
        col1,col2 = st.columns(2)
        with col1:
            profit_by_month = df.groupby(by=['MonthName'], as_index=False)['Profit'].sum().sort_values('MonthName', key=lambda x: pd.to_datetime(x, format='%B'))
            self.plot_bar_chart(st, profit_by_month, "Profit Over Month", 'MonthName', 'Profit')
        with col2:
            profit_by_category = df.groupby(by='Category', as_index=False)['Profit'].sum().sort_values(by='Profit', ascending=False).head(5)
            self.plot_pie_chart(st, profit_by_category, "Profit Distribution by Category", 'Category', 'Profit')

    # def display_correlation_analysis(self, df):
    #     """Display correlation analysis among numerical features."""
    #     st.subheader("Correlation Analysis")
    #     plt.figure(figsize=(10, 6))
    #     correlation_matrix = df.corr()
    #     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    #     st.pyplot(plt)

    def show(self):
        """Show the analysis."""
        is_valid, error_message = self.validate_schema()
        if not is_valid:
            st.error(error_message)
            return

        self.preprocess()
        self.perform_eda()
