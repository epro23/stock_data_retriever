import yfinance as yf
import pandas as pd
import plotly as py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import numpy as np
import seaborn as sns
import datetime as dt
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tkinter as tk
from tkinter import simpledialog, messagebox
import os
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- Data Fetching & Basic Manipulation --------------------
def fetch_stock_data(tickers: list, start_date, end_date, interval="60m"):
    """
    Fetch stock data for given tickers and date range.
    
    Args:
        tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date (e.g., '2024-10-01').
        end_date (str): End date (e.g., '2024-10-31').
        interval (str): Interval for data download, recommended entries: 30m, 60m, 1d, 1wk, 1mo.
    
    Returns:
        dict: Dictionary with ticker keys and their data as values, and a list of tickers.
    """
    stock_data = {}

    for ticker in tickers:
        # Download data for each ticker
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, rounding=True)
        data = data.drop(columns=["Close", "Volume"], errors="ignore")
        stock_data[f"{ticker}_data"] = data
        stock_data["tickers"] = tickers
        
    return stock_data

def correct_stock_data(stock_data):
    """
    Correct stock data by removing multi-index levels and renaming index column.
    
    Args:
        stock_data (dict): Dictionary of stock data with ticker keys.
    """
    for ticker_key, dataframe in stock_data.items():
        if ticker_key == "tickers":
            continue
        dataframe = dataframe.droplevel("Ticker", axis=1)
        dataframe.reset_index(inplace=True)
        dataframe.columns.name = "Data Index"
        if "Datetime" in dataframe.columns:
            dataframe["Datetime"] = pd.to_datetime(dataframe["Datetime"]).dt.strftime('%Y-%m-%d %H:%M')
        stock_data[ticker_key] = dataframe

    return stock_data

def format_large_number(value):
    """Helper function to format large numbers for display."""
    if value is None:
        return None
    elif value >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}K"
    else:
        return f"{value:.2f}"
    
# -------------------- Core Data Visualizations --------------------
def create_individual_graph(stock_data, file_path="individual_graphs.html"):
    """
    Generate line plots for adjusted closing prices of stocks, save them as HTML.

    Args:
        stock_data (dict): Dictionary of stock data with ticker keys.
        file_path (str): Path for saving the HTML file.

    Returns:
        dict: A dictionary containing the figures and the URL of the saved HTML file.
    """
    figures = {}

    # Generate individual figures for each stock
    for ticker, dataframe in stock_data.items():
        if ticker == "tickers":
            continue

        markers = len(dataframe) <= 62
        render_mode = "svg" if len(dataframe) > 999 else "auto"

        fig = px.line(
            dataframe,
            x=dataframe.columns[0],
            y="Adj Close",
            markers=markers,
            render_mode=render_mode,
            title=f"Price Trend for {ticker}",
        )

        # Define rangebreaks, excluding weekends and (optionally) non-trading hours
        rangebreaks = [dict(bounds=["sat", "mon"])]  # Remove weekends
        if dataframe.columns[0] == "Datetime":
            rangebreaks.append(dict(bounds=[16, 9.5], pattern="hour"))  # Exclude non-trading hours

        fig.update_layout(
            xaxis=dict(rangebreaks=rangebreaks),
            xaxis_title="Date/Time",
            yaxis_title="Adjusted Close Price",
        )

        figures[ticker] = fig

    # Handle multiple figures by combining them into a subplot
    if len(figures) > 1:
        combined_fig = make_subplots(
            rows=1,
            cols=len(figures),
            subplot_titles=[f"Price Trend for {ticker}" for ticker in figures.keys()],
        )

        for i, ticker in enumerate(figures.keys()):
            for trace in figures[ticker]["data"]:
                combined_fig.add_trace(trace, row=1, col=i + 1)

        combined_fig.update_layout(
            xaxis_title="Date/Time",
            yaxis_title="Adjusted Close Price",
        )

        # Save combined figure to HTML file
        pio.write_html(combined_fig, file=file_path, auto_open=False)
        return {"figures": figures, "html_url": file_path}

    # Handle single figure case
    first_ticker = next(iter(figures))  # Get the first (and only) ticker
    pio.write_html(figures[first_ticker], file=file_path, auto_open=False)
    return {"figures": figures, "html_url": file_path}

      
def create_news_table(stock_data, file_path="news_table.html"):
    """
    Create a table of stock news and save it as HTML file.

    Args:
        stock_data (dict): Dictionary of stock data with ticker keys.
        file_path (str): Path for saving the HTML file.

    Returns:
        dict: A dictionary containing Plotly table figure and URL of saved HTML file.
    """
    all_news = []

    # Collect news
    for ticker in stock_data["tickers"]:
        ticker_item = yf.Ticker(ticker)
        news = ticker_item.get_news()
        news_df = pd.DataFrame(news)

        # Limit the number of articles
        if len(stock_data["tickers"]) > 1:
            news_df = news_df.iloc[:3]
        else:
            news_df = news_df.iloc[:6]

        all_news.append(news_df)

    # Concat news into a single DataFrame
    final_news_df = pd.concat(all_news, ignore_index=True)
    final_news_df = final_news_df.drop(columns=["uuid", "type", "thumbnail", "link"], errors="ignore")
    final_news_df["providerPublishTime"] = pd.to_datetime(
        final_news_df["providerPublishTime"], unit="s"
    ).dt.strftime('%Y-%m-%d %H:%M')

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(final_news_df.columns),
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[final_news_df[col].tolist() for col in final_news_df.columns],
                    fill_color="whitesmoke",
                    align="left",
                ),
            )
        ]
    )

    pio.write_html(fig, file=file_path, auto_open=False)

    # Return the figure and HTML file path
    return {"figure": fig, "html_url": file_path}


def create_finances_table(stock_data, file_path="finances_table.html"):
    """
    Create a table showing financials, dispersed evenly by date into 10 rows, and save it as an HTML file.

    Args:
        stock_data (dict): Dictionary of stock data with ticker keys.
        file_path (str): Path for saving the HTML file.

    Returns:
        dict: A dictionary containing Plotly table figure and the URL of the saved HTML file.
    """
    table_subsets = {}

    # Process each ticker's DataFrame
    for ticker, dataframe in stock_data.items():
        if ticker == "tickers":
            continue

        # Reduce DataFrame to 10 rows (dispersed evenly)
        if len(dataframe) <= 10:
            table_subsets[ticker] = dataframe
        else:
            step = max(1, len(dataframe) // 9)
            reduced_df = dataframe.iloc[::step].head(9)
            reduced_df = pd.concat([reduced_df, dataframe.iloc[[-1]]])  # Include the last row

            table_subsets[ticker] = reduced_df

    # Horizontally concatenate DataFrames
    merged_table = pd.concat(table_subsets.values(), axis=1, keys=table_subsets.keys())

    # Create a Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(merged_table.columns),  # Column headers
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[merged_table[col].tolist() for col in merged_table.columns],  # Data for each column
                    fill_color="whitesmoke",
                    align="left",
                ),
            )
        ]
    )

    pio.write_html(fig, file=file_path, auto_open=False)

    # Return the figure and HTML file path
    return {"figure": fig, "html_url": file_path}


def create_important_table(stock_data, file_path="important_table.html"):
    """
    Create a secondary table displaying 'important' metrics such as beta, PE ratios, and 52-week scores,
    with tickers as columns and metrics as rows. Save the table as an HTML file.

    Args:
        stock_data (dict): Dictionary of stock data with ticker keys.
        file_path (str): Path for saving the HTML file.

    Returns:
        dict: A dictionary containing the Plotly table figure and the URL of the saved HTML file.
    """
    metrics_list = []

    # Loop through tickers and extract data
    for ticker in stock_data["tickers"]:
        ticker_item = yf.Ticker(ticker)
        metrics = {
            "Ticker": ticker,
            "Trailing P/E": round(ticker_item.info.get("trailingPE", float("nan")), 3),
            "Forward P/E": round(ticker_item.info.get("forwardPE", float("nan")), 3),
            "52 Week Low": round(ticker_item.info.get("fiftyTwoWeekLow", float("nan")), 3),
            "52 Week High": round(ticker_item.info.get("fiftyTwoWeekHigh", float("nan")), 3),
            "52 Week Change": round(ticker_item.info.get("52WeekChange", float("nan")), 3),
            "50 Day Average": round(ticker_item.info.get("fiftyDayAverage", float("nan")), 3),
            "200 Day Average": round(ticker_item.info.get("twoHundredDayAverage", float("nan")), 3),
        }
        metrics_list.append(metrics)

    # Create DataFrame and transpose for table structure
    dataframe = pd.DataFrame(metrics_list).set_index("Ticker").T

    # Create Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metrics"] + list(dataframe.columns),
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[dataframe.index.tolist()] + [dataframe[col].tolist() for col in dataframe.columns],
                    fill_color="whitesmoke",
                    align="left",
                ),
            )
        ]
    )

    # Save the table as an HTML file
    pio.write_html(fig, file=file_path, auto_open=False)

    # Return the figure and HTML file path
    return {"figure": fig, "html_url": file_path}


def create_additional_table(stock_data, file_path="additional_table.html"):
    """
    Create tertiary table to display other metrics relevant to stock purchasing decisions.

    Args:
        stock_data (dict): Dictionary of stock data with ticker keys.
        file_path (str): Path for saving the HTML file.

    Returns:
        dict: A dictionary containing the Plotly table figure and the URL of the saved HTML file.
    """
    metrics_list = []

    for ticker in stock_data["tickers"]:
        ticker_item = yf.Ticker(ticker)
        metrics = {
            "Ticker": ticker,
            "Industry": ticker_item.info.get("industry"),
            "Analyst Recommendation": ticker_item.info.get("recommendationKey"),
            "Beta": ticker_item.info.get("beta"),
            "Overall Risk Score": ticker_item.info.get("overallRisk"),
            "Market Cap": ticker_item.info.get("marketCap"),
            "Total Revenue": ticker_item.info.get("totalRevenue"),
            "Net Profit": ticker_item.info.get("netIncomeToCommon"),
            "Profit Margins": ticker_item.info.get("profitMargins"),
        }

        # Additional Calculations
        total_revenue = metrics["Total Revenue"]
        gross_margins = ticker_item.info.get("grossMargins")
        gross_profit = total_revenue * gross_margins if total_revenue and gross_margins else None
        metrics["Gross Profit"] = gross_profit

        # Format metrics for display
        if metrics["Profit Margins"] is not None:
            metrics["Profit Margins"] = f"{metrics['Profit Margins'] * 100:.2f}%"

        metrics["Market Cap"] = format_large_number(metrics["Market Cap"])
        metrics["Total Revenue"] = format_large_number(metrics["Total Revenue"])
        metrics["Net Profit"] = format_large_number(metrics["Net Profit"])
        metrics["Gross Profit"] = format_large_number(metrics["Gross Profit"])

        metrics_list.append(metrics)

    # Create DataFrame
    dataframe = pd.DataFrame(metrics_list)

    # Create Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(dataframe.columns),  # Column headers
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[dataframe[col].tolist() for col in dataframe.columns],  # Data for each column
                    fill_color="whitesmoke",
                    align="left",
                ),
            )
        ]
    )

    # Save the table as an HTML file
    pio.write_html(fig, file=file_path, auto_open=False)

    # Return the figure and HTML file path
    return {"figure": fig, "html_url": file_path}

# -------------------- Report Generation --------------------

def generate_html_report(data, output_file="final_report.html"):
    """
    Generate an HTML report by combining outputs of various functions and write it to a single file.

    Args:
        data (dict): Stock data to be passed to the individual functions.
        output_file (str): The path to save the final HTML report.

    Returns:
        str: The file path of the generated HTML report.
    """
    # Generate HTML outputs for individual sections
    html_inputs = {
        "Stock Price Chart": create_individual_graph(data)["html_url"],
        "News Table": create_news_table(data)["html_url"],
        "Finances Table": create_finances_table(data)["html_url"],
        "Important Metrics Table": create_important_table(data)["html_url"],
        "Additional Metrics Table": create_additional_table(data)["html_url"],
    }

    # Start building the HTML report
    html_report = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { text-align: center; }
            section { margin-bottom: 40px; }
            iframe { width: 100%; height: 600px; border: none; }
        </style>
    </head>
    <body>
        <h1>Stock Analysis Report</h1>
    """

    # Add each HTML file as an iframe
    for idx, (title, html_file) in enumerate(html_inputs.items(), 1):
        html_report += f"""
        <section>
            <h2>{title}</h2>
            <iframe src="{html_file}"></iframe>
        </section>
        """

    # Close the HTML tags
    html_report += """
    </body>
    </html>
    """

    # Write the final HTML report to a file
    with open(output_file, "w") as file:
        file.write(html_report)

    return output_file

# -------------------- Final GUI Workflow --------------------
def gui_workflow():
    """
    GUI workflow to compile all functions and generate the final HTML report.
    """
    # Create root window, hide it
    root = tk.Tk()
    root.withdraw()

    # Step 1: Get user input through dialogs
    tickers = simpledialog.askstring("Input Tickers", "Enter stock tickers (comma-separated):")
    start_date = simpledialog.askstring("Input Start Date", "Enter start date (YYYY-MM-DD):")
    end_date = simpledialog.askstring("Input End Date", "Enter end date (YYYY-MM-DD):")
    interval = simpledialog.askstring("Input Time Interval", "Enter time interval (30m, 60m, 1d, 1wk, 1mo) ")

    # Step 2: Fetch and process data
    raw_data = fetch_stock_data(tickers.split(","), start_date, end_date)
    data = correct_stock_data(raw_data)

    # Step 3: Generate HTML report
    output_path = "final_report.html"
    report_path = generate_html_report(data, output_file=output_path)

    # Display success message
    messagebox.showinfo("Success", f"HTML report generated successfully: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    gui_workflow()