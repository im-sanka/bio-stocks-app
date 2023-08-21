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
import yahooquery as yq
from yahooquery import Ticker

def financial_statements_eda(ticker_symbol):
    # Fetch general stock details
    info = Ticker(ticker_symbol).summary_detail[ticker_symbol]

    # Calculate EPS
    eps = None
    if 'previousClose' in info and 'trailingPE' in info and info['trailingPE'] != 0:
        eps = info['previousClose'] / info['trailingPE']

    # Fetch data
    ticker = yq.Ticker(ticker_symbol)

    # Balance Sheet
    balance_sheet = ticker.balance_sheet(frequency="q")
    balance_sheet = balance_sheet.set_index('asOfDate').sort_index()
    # balance_sheet.columns
    # Cash Flow
    cash_flow = ticker.cash_flow(frequency="q")
    cash_flow = cash_flow.set_index('asOfDate').sort_index()
    # cash_flow.columns
    # Income Statement
    income_statement = ticker.income_statement(frequency="q")
    income_statement = income_statement.set_index('asOfDate').sort_index()
    # st.write(income_statement.columns)

    # Create subplots
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Balance Sheet", "Cash Flow", "Income Statement"))

    # Add traces for Balance Sheet
    fig.add_trace(go.Scatter(x=balance_sheet.index, y=balance_sheet['TotalAssets'], mode='lines+markers', name='Total Assets'), row=1, col=1)
    fig.add_trace(go.Scatter(x=balance_sheet.index, y=balance_sheet['TotalLiabilitiesNetMinorityInterest'], mode='lines+markers', name='Total Liabilities NMI'), row=1, col=1)

    # Add traces for Cash Flow
    fig.add_trace(go.Scatter(x=cash_flow.index, y=cash_flow['OperatingCashFlow'], mode='lines+markers', name='Operating Cash Flow'), row=2, col=1)
    fig.add_trace(go.Scatter(x=cash_flow.index, y=cash_flow['InvestingCashFlow'], mode='lines+markers', name='Investing Cash Flow'), row=2, col=1)

    # Add traces for Income Statement
    fig.add_trace(go.Scatter(x=income_statement.index, y=income_statement['TotalRevenue'], mode='lines+markers', name='Total Revenue'), row=3, col=1)
    fig.add_trace(go.Scatter(x=income_statement.index, y=income_statement['NetIncome'], mode='lines+markers', name='Net Income'), row=3, col=1)

    # Set layout
    fig.update_layout(
        title_text=f'Financial Statements for {ticker_symbol}',
        template="plotly_dark", height=900,
        legend_title="Legend",
        legend=dict(x=1.0, y=0.5),
    )

    # Extracting necessary values for ratios
    net_income = income_statement['NetIncome'].iloc[-1]  # Most recent quarter's net income
    total_debt = balance_sheet['TotalLiabilitiesNetMinorityInterest'].iloc[-1]  # Most recent quarter's total debt
    beginning_equity = balance_sheet['TotalAssets'].iloc[0] - balance_sheet['TotalLiabilitiesNetMinorityInterest'].iloc[0]
    ending_equity = balance_sheet['TotalAssets'].iloc[-1] - balance_sheet['TotalLiabilitiesNetMinorityInterest'].iloc[-1]
    average_equity = (beginning_equity + ending_equity) / 2

    # Calculating Debt/Equity Ratio
    if total_debt and average_equity:  # Ensure values are not zero to avoid ZeroDivisionError
        debt_equity_ratio = total_debt / average_equity
    else:
        debt_equity_ratio = 'N/A'

    # Calculating Return on Equity (RoE)
    if net_income and average_equity:  # Ensure values are not zero to avoid ZeroDivisionError
        roe_ratio = net_income / average_equity
    else:
        roe_ratio = 'N/A'

    pe_ratio = info.get('trailingPE', 'N/A')

    col1, col2, col3, col4 = st.columns(4)
    col1.subheader("Debt/Equity Ratio:")
    col1.write(f"<h6 style='font-size:80px'>{debt_equity_ratio:.2f}</h6>", unsafe_allow_html=True)

    col2.subheader("RoE Ratio:")
    col2.write(f"<h6 style='font-size:80px'>{roe_ratio:.2f}</h6>", unsafe_allow_html=True)

    col3.subheader("EPS:")
    col3.write(f"<h6 style='font-size:80px'>{round(eps, 2) if eps is not None else 'N/A'}</h6>", unsafe_allow_html=True)

    col4.subheader("P/E Ratio (Trailing)")
    col4.write(f"<h6 style='font-size:80px'>{round(pe_ratio, 2) if pe_ratio != 'N/A' else 'N/A'}</h6>", unsafe_allow_html=True)

    def evaluate_stock(debt_equity, roe, eps):
        evaluations = []

        # Evaluate Debt/Equity
        if debt_equity < 0.5:
            evaluations.append("Low leverage **(debt equity < 0.5)**: More financially stable.")
        elif 0.5 <= debt_equity < 1:
            evaluations.append("Moderate leverage **(0.5 <= debt_equity < 1)**: Typical for many companies.")
        else:
            evaluations.append("High leverage **(debt_equity > 1)**: Risky, especially in economic downturns.")

        # Evaluate RoE
        if roe > 15:
            evaluations.append("High Return on Equity **(RoE > 15)**: Company effectively uses its equity.")
        elif 10 < roe <= 15:
            evaluations.append("Moderate Return on Equity. **(10 < RoE <= 15)**")
        else:
            evaluations.append("Low Return on Equity **(RoE < 15)**: Profitability or leverage issues.")

        # Evaluate EPS
        if eps:
            evaluations.append("**Positive EPS**: Company is profitable.")
        else:
            evaluations.append("**EPS data is not available or negative value** indicates potential future growth or there is an issue in current profitability.")
        # Evaluate P/E Ratio
        col1,col2,col3 = st.columns(3)
        industry_average_pe = col1.number_input("Please input your five-year projected growth rate (the default is 12.8):", 12.8)
        try:
            if pe_ratio < industry_average_pe:
                evaluations.append(f"**P/E below projected growth rate ({industry_average_pe}%)**: Potentially undervalued.")
            elif pe_ratio > industry_average_pe:
                evaluations.append(f"**P/E above projected growth rate ({industry_average_pe}%)**: Potentially overvalued.")
        except:
            evaluations.append(f"**P/E ratio** is not available (N/A)")
        return evaluations

    with st.expander("**Explanation About Financial Metrics**"):
        st.markdown("""
        - **Debt/Equity Ratio**: Represents a company's financial leverage. It's the proportion of equity and debt a company is using to finance its assets. A high ratio suggests that a company has aggressively financed its growth with debt.
        - **Return on Equity (RoE)**: Measures a company's profitability by revealing how much profit a company generates with the money shareholders have invested.
        - **EPS (Earnings Per Share)**: Represents the portion of a company's profit allocated to each outstanding share of common stock.
        - **P/E Ratio (Trailing)**: Represents the valuation ratio of a company's current share price compared to its per-share earnings over the past 12 months.
        """)

    with st.expander("Insights! __not promoting/ taking any responsibility regarding your action!__"):
        evaluations = evaluate_stock(debt_equity_ratio, roe_ratio, eps)
        for eval in evaluations:
            st.write(eval)

    return fig

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
    trend = round(trend*100,2)

    return trend  # Convert trend to percentage

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
st.title("Stock Data Dashboard :chart_with_upwards_trend::bar_chart:")

# User sidebar input for tickers
manual_tickers = st.sidebar.text_input("Enter stock tickers separated by a comma, for examples: TSLA, GOOG, AAPL, etc.", "").split(", ")

# Upload option for tickers file
uploaded_file = st.sidebar.file_uploader("Or upload a file with stock tickers", type=["txt"])

if uploaded_file:
    uploaded_tickers = process_uploaded_file(uploaded_file)
    stock_symbols = list(set(manual_tickers + uploaded_tickers))
    stock_symbols = [(ticker, "Unknown") for ticker in stock_symbols]  # Assuming the uploaded file only has tickers
elif manual_tickers and manual_tickers[0]:
    stock_symbols = manual_tickers
    stock_symbols = [(ticker, "Unknown") for ticker in stock_symbols]  # Assuming manual tickers only have tickers
else:
    with open("stocks_list.txt", "r") as file:
        stock_symbols = [tuple(line.strip().split(" - ")) for line in file]  # This gives [(ticker, company_name), ...]

download_link = get_file_download_link("stocks_list.txt", "the list example here!")
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
    selected_stock = st.selectbox("Select a stock symbol:", [f"{s[0]} - {s[1]}" for s in stock_symbols]).split(" - ")[0]
    data = fetch_data(selected_stock, start_date, end_date)

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    data['bb_h'] = bb_indicator.bollinger_hband()
    data['bb_l'] = bb_indicator.bollinger_lband()
    # MACD
    data['macd'] = MACD(data.Close).macd()
    data['macd_signal'] = MACD(data.Close).macd_signal()
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
    if st.checkbox("Show Moving average convergence/divergence (MACD) - __recommendation: use this without ticking other indicators.__"):
        fig.add_trace(go.Scatter(x=data.index, y=data["macd"], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=data["macd_signal"], mode='lines', name='Signal-MACD'))
    if st.checkbox("Show Relative strength index (RSI)"):
        fig.add_trace(go.Scatter(x=data.index, y=data["rsi"], mode='lines', name='RSI'))
        fig.add_trace(go.Scatter(x=data.index, y=[30] * len(data.index), mode='lines', name='RSI=30', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=[70] * len(data.index), mode='lines', name='RSI=70', line=dict(dash='dash')))
    if st.checkbox("Show Simple moving average (SMA) - fixed window"):
        fig.add_trace(go.Scatter(x=data.index, y=data["sma"], mode='lines', name='SMA'))
    if st.checkbox("Show Exponential moving average (EMA)"):
        fig.add_trace(go.Scatter(x=data.index, y=data["ema"], mode='lines', name='EMA'))
    if st.checkbox("Show Adjustable Moving Average"):
        for window in selected_ma_windows:
            fig.add_trace(go.Scatter(x=data.index, y=data[f"MA{window}"], mode='lines', name=f"MA{window}"))

    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], yaxis="y2", name="Volume", opacity=0.6, marker=dict(color='lightblue')))
    fig.update_layout(title=f"{selected_stock} Indicators Over Time",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      yaxis2=dict(overlaying='y', side='right', title="Volume"),
                      legend_title="Legend",
                      legend=dict(x=1.10, y=0.5),
                      height=800)

    with st.expander("__Explanation about indicators:__"):
        st.markdown("""
            - Candlestick chart indicates the fluctuation of stock price each day.
            - Close history provides a line from each day stock price after closing.
            - Bollinger bands show overbought (upper)/ oversold (lower) of the stocks.
            - Moving average convergence/divergence (MACD) line crosses from below to above the signal line = bullish (uptrend). Value less than zero line gives stronger signals.
            - Relative strength index (RSI) indicates buy/sell signal. High (>70) can indicate bearish signal while low (<30) indicates bullish signal.
            - Simple moving average (SMA) shows a moving average (14 datapoints/prices) and put equal weight.
            - Exponential moving average (EMA) is a moving average (14 datapoints/prices) and gives exponential weight to current/recent data.
            ---""")
    st.plotly_chart(fig, use_container_width=True)




    eda_fig = financial_statements_eda(selected_stock)
    st.plotly_chart(eda_fig, use_container_width=True)

elif view_option == "Advanced":
    # st.header("Stock Comparison")
    selected_options = st.multiselect(
        'Select stocks for comparison:',
        [f"{s[0]} - {s[1]}" for s in stock_symbols]
    )
    selected_stocks = [stock.split(' - ')[0] for stock in selected_options]
    # selected_stocks = [stock.split(" - ")[0] for stock in st.multiselect("Select stocks for comparison:", stock_symbols)]

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

    eda_data = pd.DataFrame(eda_data).sort_values(by="Trend (%)", ascending=False).reset_index(drop=True)
    eda_data['Trend (%)'] = eda_data['Trend (%)'].apply(lambda x: round(x, 2))

    ticker_to_fullname = {s[0]: f"{s[0]} - {s[1]}" for s in stock_symbols}
    eda_data["Stock"] = eda_data["Stock"].map(ticker_to_fullname)

    col1.write(eda_data)

    # Displaying the correlation matrix
    col2.subheader("Correlation matrix between selected stocks")
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

        annotations = []
        for i, row in enumerate(mask.values):
            for j, value in enumerate(row):
                if not np.isnan(value):
                    annotations.append({
                        "x": mask.columns[j],
                        "y": mask.columns[i],
                        "xref": "x",
                        "yref": "y",
                        "text": str(round(value, 2)),  # You can format this as you see fit
                        "showarrow": False,
                        "font": {
                            "color": "black"  # You can adjust this for better visibility based on your colorscale
                        }
                })

        fig_corr = go.Figure(go.Heatmap(z=mask, x=mask.columns, y=mask.columns, colorscale='Sunset', zmin=-1, zmax=1))

        fig_corr.update_layout(title="This matrix shows 'Close' data from selected stocks",
                               margin=dict(t=50), width=400, height=800,
                               annotations=annotations)  # Adjust width and height as needed
        col2.plotly_chart(fig_corr, use_container_width=True)

# Prediction
elif view_option == "Analytic/ Prediction":
    st.subheader("Will be updated soon! :smile: ")


if __name__ == "__main__":
    st.markdown("What do you think about this? Let me know your comments! [E-mail me here!](mailto:immanuel.sanka@gmail.com)")
