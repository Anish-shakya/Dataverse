import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error


class NepseStockPrediction:
    
    @staticmethod
    def load_lstm_model(path):
        """Load the LSTM model from the specified path."""
        return load_model(path)

    @staticmethod
    def load_stock_data(path):
        """Load the stock data from a CSV file."""
        return pd.read_csv(path)

    @staticmethod
    def filter_stock_data(df, selected_scrip):
        """Filter the stock data based on the selected scrip."""
        return df[df['Scrip'].isin(selected_scrip)].reset_index(drop=True)

    @staticmethod
    def calculate_moving_averages(df):
        """Calculate moving averages for the stock data."""
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        return df

    @staticmethod
    def preprocess_data(df):
        """Preprocess the stock data for LSTM model prediction."""
            # Convert the 'Close' column to numeric, forcing any invalid data to NaN
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        # Drop rows with missing values (NaN)
        df = df.dropna(subset=['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        # Creating a data structure with 60 time steps and 1 output
        X_test = []
        for i in range(60, len(scaled_data)):
            X_test.append(scaled_data[i-60:i, 0])
        
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape for LSTM input
        
        return X_test, scaler

    @staticmethod
    def predict_prices(model, X_test, scaler):
        """Use the LSTM model to predict stock prices."""
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)  # Inverse scaling
        
        return predicted_stock_price
    
    @staticmethod
    def predict_future_prices(model, last_60_days_data, scaler, days_to_predict=10):
        """Predict future stock prices for a given number of days."""        
        # Prepare input data for future predictions
        scaled_data = scaler.transform(last_60_days_data.reshape(-1, 1))
        future_X = np.array([scaled_data[-60:]])
        future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))

        future_predictions = []
        for _ in range(days_to_predict):
            future_pred = model.predict(future_X)
            future_predictions.append(future_pred[0, 0])
            
            # Update input data with the latest prediction
            future_X = np.append(future_X[:, 1:, :], [[future_pred]], axis=1)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions

    @staticmethod
    def plot_chart(df, scrip, columns, title, colors):
        """General function to plot stock charts with moving averages."""
        fig = go.Figure()

        for i, column in enumerate(columns):
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[column],
                mode='lines', name=f'{scrip} {column}',
                line=dict(color=colors[i])
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark'
        )

        st.plotly_chart(fig)

    @staticmethod
    def plot_original_vs_predicted_chart(predicted_df, scrip):
        """Plot a chart comparing original and predicted stock prices with only the last 10% of predicted data."""
        fig = go.Figure()

        # Original closing prices
        fig.add_trace(go.Scatter(
            x=predicted_df['Date'], y=predicted_df['Original Close'],
            mode='lines', name=f'{scrip} Original Close',
            line=dict(color='green')
        ))

        # Predicted closing prices - only last 10% of the predictions
        last_10_percent_index = int(len(predicted_df) * 0.1)
        predicted_df_last_10 = predicted_df.iloc[-last_10_percent_index:]

        fig.add_trace(go.Scatter(
            x=predicted_df_last_10['Date'], y=predicted_df_last_10['Predicted Close'],
            mode='lines', name=f'{scrip} Predicted Close (Last 10%)',
            line=dict(color='red', width=1.5)  # Dashed line for prediction
        ))

        # Update the layout
        fig.update_layout(
            title=f'{scrip} Original vs Predicted Closing Prices (Last 10% of Predictions)',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark'
        )

        # Display the chart
        st.plotly_chart(fig)
    @staticmethod
    def plot_full_original_vs_predicted_chart(predicted_df, scrip):
        """Plot a chart comparing original and predicted stock prices for the full dataset."""        
        fig = go.Figure()

        # Original closing prices
        fig.add_trace(go.Scatter(
            x=predicted_df['Date'], y=predicted_df['Original Close'],
            mode='lines', name=f'{scrip} Original Close',
            line=dict(color='green')
        ))

        # Predicted closing prices
        fig.add_trace(go.Scatter(
            x=predicted_df['Date'], y=predicted_df['Predicted Close'],
            mode='lines', name=f'{scrip} Predicted Close',
            line=dict(color='red',width=1)  # Solid line for prediction
        ))

        # Update the layout
        fig.update_layout(
            title=f'{scrip} Original vs Predicted Closing Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark'
        )

        # Display the chart
        st.plotly_chart(fig)
    
    @staticmethod
    def predict_future_prices(model, last_60_days_data, scaler, days_to_predict=10):
        """Predict future stock prices for a given number of days."""
        # Prepare input data for future predictions
        scaled_data = scaler.transform(last_60_days_data.reshape(-1, 1))
        future_X = np.array([scaled_data[-60:]])
        future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))

        future_predictions = []
        for _ in range(days_to_predict):
            future_pred = model.predict(future_X)
            future_predictions.append(future_pred[0, 0])
            
            # Update input data with the latest prediction
            # Ensure future_pred is reshaped correctly to match future_X dimensions
            future_pred = future_pred.reshape(1, 1, 1)
            future_X = np.append(future_X[:, 1:, :], future_pred, axis=1)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions

    @staticmethod
    def display_stock_charts(df, scrip):
        """Display various stock charts for the selected scrip."""
        # 1. Plot Stock Closing Price History
        NepseStockPrediction.plot_chart(
            df, scrip, ['Close'], f'{scrip} Stock Closing Price', ['blue'])

        # 2. Plot Stock Closing Price with MA50
        NepseStockPrediction.plot_chart(
            df, scrip, ['Close', 'MA50'], f'{scrip} Stock Closing Price vs MA50', ['blue', 'red'])

        # 3. Plot Stock Closing Price with MA50 and MA100
        NepseStockPrediction.plot_chart(
            df, scrip, ['Close', 'MA50', 'MA100'], f'{scrip} Stock Closing Price vs MA50 vs MA100', ['blue', 'red', 'green'])

        # 4. Plot Stock Closing Price with MA100 and MA200
        NepseStockPrediction.plot_chart(
            df, scrip, ['Close', 'MA100', 'MA200'], f'{scrip} Stock Closing Price vs MA100 vs MA200', ['blue', 'green', 'purple'])
        
    @staticmethod
    def display_model_summary(model):
        """Display model summary in Streamlit."""
        from io import StringIO
        string_io = StringIO()
        model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        summary_string = string_io.getvalue()
        st.subheader("LSTM Model Summary")
        st.text(summary_string)
        
    @staticmethod
    def calculate_metrics(y_test, predicted_stock_price):
        """Calculate and return MAE, MAPE, RMSE and Accuracy."""
        # MAE: Mean Absolute Error
        mae = mean_absolute_error(y_test, predicted_stock_price)
        
        # MAPE: Mean Absolute Percentage Error (in percentage)
        mape = np.mean(np.abs((y_test - predicted_stock_price) / y_test)) * 100
        
        # RMSE: Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
        
        # Accuracy: As 100 - MAPE (rough approximation)
        accuracy = 100 - mape
        
        return mae, mape, rmse, accuracy

    @staticmethod
    def prediction():
        # Load the LSTM model
        model_path = 'D:\\College Files\\8th Semester\\Project III\\Dataverse\\Pages\\Prediction\\Models\\NepseModel.keras'
        model = NepseStockPrediction.load_lstm_model(model_path)
        
        # Display header
        st.header('Nepse Stock Prediction Using LSTM📊')

        # Load the Nepse stock dataset
        data_path = 'D:\\College Files\\8th Semester\\Project III\\Dataverse\\Datasets\\NepseStock.csv'
        df = NepseStockPrediction.load_stock_data(data_path)

        # Get unique stock tickers (scrips)
        Stock_Ticker = df['Scrip'].unique()

        # Allow the user to select one or more scrips
        selected_scrip = st.multiselect('Select Nepse Stock Scrip', Stock_Ticker)

        if not selected_scrip:
            # If no scrip is selected, show the first 10 rows of the full dataset
            st.write(df.head(10))
        else:
            # Filter the DataFrame based on the selected scrip(s)
            df_filtered = NepseStockPrediction.filter_stock_data(df, selected_scrip)

            # Show the filtered dataset (Optional)
            st.write(df_filtered.head(10))

            # Iterate through the selected scrips
            for scrip in selected_scrip:
                filtered_df = df_filtered[df_filtered['Scrip'] == scrip]

                # Calculate moving averages for the filtered data
                filtered_df = NepseStockPrediction.calculate_moving_averages(filtered_df)

                # Display various stock charts for the selected scrip
                NepseStockPrediction.display_stock_charts(filtered_df, scrip)

                # Preprocess data for LSTM prediction
                X_test, scaler = NepseStockPrediction.preprocess_data(filtered_df)

                # Predict stock prices using the LSTM model
                predicted_stock_price = NepseStockPrediction.predict_prices(model, X_test, scaler)

                # Create a new DataFrame for the predicted stock prices
                # Align the predicted values with the original data's date range
                predicted_df = pd.DataFrame({
                    'Date': filtered_df['Date'][-len(predicted_stock_price):],
                    'Original Close': filtered_df['Close'][-len(predicted_stock_price):],
                    'Predicted Close': predicted_stock_price.flatten()  # Flatten the predicted values
                })

                # Plot the "Original vs Predicted" chart
                NepseStockPrediction.plot_original_vs_predicted_chart(predicted_df, scrip)

                 # Plot the "Original vs Predicted" chart for the full dataset
                NepseStockPrediction.plot_full_original_vs_predicted_chart(predicted_df, scrip)
                
    @staticmethod
    def show():
        # Load the LSTM model
        model_path = 'D:\\College Files\\8th Semester\\Project III\\Dataverse\\Pages\\Prediction\\Models\\NepseModel.keras'
        model = NepseStockPrediction.load_lstm_model(model_path)
        
        # Call the prediction method (to calculate and display metrics)
        NepseStockPrediction.prediction()

         # Display model summary
         # Display the metrics in Streamlit
        st.header("Model Evaluation Metrics 📊")
        NepseStockPrediction.display_model_summary(model)
        #Your calculated metrics
        mae = 42.08198696501357
        mape = 1.9186218957088508
        rmse = 5.545821289062497
        r2 = 0.9658606410805906
        st.subheader("Evaluation Results:")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R-squared (Accuracy):** {r2:.2f}")
            
