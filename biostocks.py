import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import datetime
import pytz
import base64
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator

# Fetch stock data function
def fetch_data(symbol, start_date=None, end_date=None):
    return yf.Ticker(symbol).history(start=start_date, end=end_date)

def normalize_data(df):
    """Normalize stock data to the range [0, 1] based on its history."""
    return (df - df.min()) / (df.max() - df.min())

def calculate_trend(data, start_date, end_date):
    """Calculate percentage trend over the given date range."""
    start_date_tz = pytz.timezone('America/New_York').localize(datetime.datetime.combine(start_date, datetime.time()))
    end_date_tz = pytz.timezone('America/New_York').localize(datetime.datetime.combine(end_date, datetime.time()))
    closest_start = data.index[data.index >= start_date_tz].min()
    closest_end = data.index[data.index <= end_date_tz].max()

    if pd.isna(closest_start) or pd.isna(closest_end):
        return "Not enough data"

    trend = (data.loc[closest_end, "Close"] - data.loc[closest_start, "Close"]) / data.loc[closest_start, "Close"]
    return trend * 100  # Convert trend to percentage

def process_uploaded_file(uploaded_file):
    try:
        tickers = uploaded_file.read().decode("utf-8").splitlines()
        return [ticker.split(" - ")[0] for ticker in tickers]
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return []

def get_file_download_link(file_path, filename):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">Download {filename}</a>'

# Streamlit app
st.set_page_config(layout="wide")

st.image("logo.png")
st.title("Stock Data Dashboard")

# User sidebar input for tickers
manual_tickers = st.sidebar.text_input("Enter stock tickers separated by a comma, for examples: TSLA, GOOG, AAPL, etc.", "").split(", ")

# Upload option for tickers file
uploaded_file = st.sidebar.file_uploader("Or upload a file with stock tickers", type=["txt"])

if uploaded_file:
    uploaded_tickers = process_uploaded_file(uploaded_file)
    stock_symbols = list(set(manual_tickers + uploaded_tickers))
elif manual_tickers and manual_tickers[0]:
    stock_symbols = manual_tickers
else:
    with open("../stocks_list.txt", "r") as file:
        stock_symbols = [line.strip().split(" - ")[0] for line in file]

download_link = get_file_download_link("../stocks_list.txt", "the list example here!")
st.sidebar.markdown(download_link, unsafe_allow_html=True)

# Date Range Selector
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Option to switch between "basic" and "advanced" views in the sidebar
view_option = st.sidebar.selectbox("Choose view:", ["Basic", "Advanced","Analytic/ Prediction"])

# Moving average option
default_ma_windows = [1, 3, 5, 7, 10, 20, 60, 120]
selected_ma_windows = st.sidebar.multiselect("Select window sizes for moving averages:", default_ma_windows, default=default_ma_windows)

if view_option == "Basic":
    # st.header("Single Stock Exploration")

    selected_stock = st.selectbox("Select a stock symbol:", stock_symbols).split(" - ")[0]
    data = fetch_data(selected_stock, start_date, end_date)

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    data['bb_h'] = bb_indicator.bollinger_hband()
    data['bb_l'] = bb_indicator.bollinger_lband()
    # MACD
    data['macd'] = MACD(data.Close).macd()
    # RSI
    data['rsi'] = RSIIndicator(data.Close).rsi()
    # SMA
    data['sma'] = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    data['ema'] = EMAIndicator(data.Close).ema_indicator()

    for window in selected_ma_windows:
        data[f"MA{window}"] = data["Close"].rolling(window=window).mean()

    fig = go.Figure()

    # Add the Candlestick chart if the checkbox is checked
    if st.checkbox("Show Candlestick Chart"):
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     increasing_line_color='green',
                                     decreasing_line_color='red'))

    # Checkboxes for other data plots
    if st.checkbox("Show Close history"):
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close'))
    if st.checkbox("Show Bollinger Bands"):
        fig.add_trace(go.Scatter(x=data.index, y=data["bb_h"], mode='lines', name='Upper Band', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data["bb_l"], mode='lines', name='Lower Band', line=dict(dash='dash')))
    if st.checkbox("Show MACD"):
        fig.add_trace(go.Scatter(x=data.index, y=data["macd"], mode='lines', name='MACD'))
    if st.checkbox("Show RSI"):
        fig.add_trace(go.Scatter(x=data.index, y=data["rsi"], mode='lines', name='RSI'))
    if st.checkbox("Show SMA"):
        fig.add_trace(go.Scatter(x=data.index, y=data["sma"], mode='lines', name='SMA'))
    if st.checkbox("Show EMA"):
        fig.add_trace(go.Scatter(x=data.index, y=data["ema"], mode='lines', name='EMA'))
    if st.checkbox("Show Moving Average"):
        for window in selected_ma_windows:
            fig.add_trace(go.Scatter(x=data.index, y=data[f"MA{window}"], mode='lines', name=f"MA{window}"))

    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], yaxis="y2", name="Volume", opacity=0.6))
    fig.update_layout(title=f"{selected_stock} Indicators Over Time",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      yaxis2=dict(overlaying='y', side='right', title="Volume"),
                      legend_title="Legend",
                      legend=dict(x=1.15, y=0.5),
                      height=800)

    st.plotly_chart(fig, use_container_width=True)

elif view_option == "Advanced":
    # st.header("Stock Comparison")

    selected_stocks = [stock.split(" - ")[0] for stock in st.multiselect("Select stocks for comparison:", stock_symbols)]

    comparison_choices = ["Open", "Close", "Margin (High - Low)"] + [f"MA{window}" for window in selected_ma_windows]
    comparison_choice = st.selectbox("Comparison data:", comparison_choices)

    normalize = st.checkbox("Normalize Data")

    # Adjusting the layout for three plots
    subplot_titles = (f"Comparison based on {comparison_choice}", "Volume Comparison", "Volatility")
    fig = make_subplots(rows=3, cols=1, subplot_titles=subplot_titles, shared_xaxes=True, vertical_spacing=0.1)

    eda_data = {"Stock": [], "Trend (%)": []}

    # Setting up a color map for the stocks
    colors = px.colors.qualitative.Plotly
    color_map = {stock: colors[i % len(colors)] for i, stock in enumerate(selected_stocks)}

    for stock in selected_stocks:
        comp_data = fetch_data(stock, start_date, end_date)

        # Computing volatility
        comp_data['volatility'] = comp_data['Close'].rolling(window=7).std()

        if comparison_choice == "Margin (High - Low)":
            comp_data["Margin"] = comp_data["High"] - comp_data["Low"]
            comparison_data = "Margin"
        elif comparison_choice.startswith("MA"):
            window = int(comparison_choice[2:])
            comp_data[f"MA{window}"] = comp_data["Close"].rolling(window=window).mean()
            comparison_data = f"MA{window}"
        else:
            comparison_data = comparison_choice

        if normalize:
            comp_data[comparison_data] = normalize_data(comp_data[comparison_data])

        stock_color = color_map[stock]

        # Adding traces for all plots in the subplot
        fig.add_trace(go.Scatter(x=comp_data.index, y=comp_data[comparison_data], mode='lines', name=stock, line=dict(color=stock_color)), row=1, col=1)
        fig.add_trace(go.Bar(x=comp_data.index, y=comp_data["Volume"], name=stock, opacity=0.6, marker_color=stock_color, showlegend=False), row=2, col=1)  # Set showlegend to False for repeated legends
        fig.add_trace(go.Scatter(x=comp_data.index, y=comp_data['volatility'], mode='lines', name=f"Volatility {stock}", opacity=0.6, line=dict(color=stock_color), showlegend=False), row=3, col=1)  # Set showlegend to False for repeated legends

        trend = calculate_trend(comp_data, start_date, end_date)
        eda_data["Stock"].append(stock)
        eda_data["Trend (%)"].append(trend)

    fig.update_layout(yaxis_title=comparison_choice,
                      legend_title="Legend",
                      legend=dict(x=1.1, y=1),
                      height=800)

    st.plotly_chart(fig, use_container_width=True)

    col1,col2 = st.columns(2)
    col1.subheader("Stock Trends")
    col1.write("This stock trends are based on the start and end dates.")
    col1.write("So, this is the __overall difference__.")
    col1.table(pd.DataFrame(eda_data))

    # Displaying the correlation matrix
    col2.subheader("Correlation Matrix of Stocks")
    # Correlation Matrix
    if len(selected_stocks) > 1:
        dataframes = []
        for stock in selected_stocks:
            df = fetch_data(stock, start_date, end_date)[["Close"]]
            df = df.rename(columns={"Close": stock})
            dataframes.append(df)

        merged_df = pd.concat(dataframes, axis=1)
        correlation_matrix = merged_df.corr()

        # Making the upper triangle of the matrix to have NaN values
        mask = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape)).astype(bool))

        fig_corr = go.Figure(go.Heatmap(z=mask, x=mask.columns, y=mask.columns, colorscale='Inferno', zmin=-1, zmax=1))

        fig_corr.update_layout(title="This matrix shows the correlation in 'Close' data between stocks.",
                               margin=dict(t=50), width=400, height=400)  # Adjust width and height as needed
        col2.plotly_chart(fig_corr, use_container_width=True)

# Prediction
elif view_option == "Analytic/ Prediction":
    st.subheader("Will be updated soon! :smile: ")


if __name__ == "__main__":
    st.markdown("What do you think about this? Let me know your comments! [E-mail me here!](mailto:immanuel.sanka@gmail.com)")
