import streamlit as st
from pandas import DataFrame
from numpy import square, mean
import yahoo_fin.stock_info as si
from datetime import datetime
import matplotlib.pyplot as plt 

st.set_page_config(
    page_title="Beta Calculator",
    page_icon="🔎",
    )

from utils import functions

header = st.container()
UserInput = st.container()
dataset = st.container()
lineChart = st.container()
results = st.container()
params = st.container()

with header:
    st.title("Beta Calculation Program")

with params:
    marketOptions = ["S&P 500 Index", "NASDAQ-100 Index", "Russell 1000 Index"]

with UserInput:
    st.header("Please Provide Information Needed")
    st.markdown("Choose a stock and date range, please")
    marketSelection = st.selectbox("Select the market:", options = marketOptions)
    stockDict = functions.stocks(marketSelection)
    firstCol, secCol = st.columns(2)
    # Requires user selection from the stock list
    choosenStock = firstCol.selectbox("Stock", options=stockDict.keys())
    # Displays ticker name of the selected stock
    choosenTicker = stockDict[choosenStock]
    # Assigns one year ago as default beginning date
    defaultBegDate= datetime(datetime.now().year-1, datetime.now().month, datetime.now().day)
    # Calender components for user's date entry
    startDate = firstCol.date_input("Pick a start date ",value= defaultBegDate, max_value= datetime.now ())
    secCol._text_input("Ticker", choosenTicker, disabled=True)
    endDate = secCol.date_input("Pick an end date", min_value= startDate, max_value= datetime.now ())

with dataset:

    # Calculates daily return of the selected stock share
    dfs = si.get_data(choosenTicker, start_date = startDate, end_date = endDate)
    dfs["daily return"] = dfs["adjclose"].pct_change()
    dfs["daily return"] = dfs["daily return"]*100
    # Calculates daily return of the market index
    dfm = si.get_data("NDX", start_date = startDate, end_date = endDate)
    dfm["daily return"] = dfm["adjclose"].pct_change()
    dfm["daily return"] = dfm["daily return"]*100

with lineChart:
    st.header("Line Charts")
    st.subheader(f"{choosenStock} Line Chart")
    st.line_chart(dfs["adjclose"])

    st.subheader(f"{choosenStock} Market Relative Line Chart")
    fig, ax = plt.subplots()
    ax.plot((dfs["adjclose"] / dfs["adjclose"].iloc[0]*100))
    ax.plot((dfm["adjclose"] / dfm["adjclose"].iloc[0]*100))
    st.pyplot(fig)

with results:
# Checks for compatibility of data in regard of dataframe length
    st.header("Decomposition of Returns")
    if dfs.shape[0] == dfm.shape[0]:
        index = list(range(dfs.shape[0]))
        n = dfs.shape[0] - 1
        # Defines the table of Decomposition of Returns for the Single-Index Model
        returnsTable = {'Index': index, 'Return on Stock': dfs["daily return"], 'Return on Market': dfm["daily return"]}
        returnsTableDf = DataFrame(data=returnsTable)
        # save button ile kullanıcıya kaydetme imkanı verilmeli. returnsTableDf.to_csv(path)
        st.write(returnsTableDf[1:])
        st.markdown("The historical data is feed via Yahoo Finance.")
        # Calculation of the Variance of the Return on Market and Return on Stock
        meanReturnMarket = mean(returnsTableDf["Return on Market"][1:])
        meanReturnStock = mean(returnsTableDf["Return on Stock"][1:])
        # Defines the variable for the sum of Return on Market minus Mean
        sumVarMarket = 0
        # Defines the variable of the product of the Return on Market minus Mean and Return on Stock minus Mean
        sumProduct = 0
        i = 1
        while i <=n:
            sumVarMarket += square(returnsTableDf.iloc[i][2] - meanReturnMarket)
            sumProduct += (returnsTableDf.iloc[i][1] - meanReturnStock)*(returnsTableDf.iloc[i][2] - meanReturnMarket)
            i +=1
        varReturnMarket = sumVarMarket/n         # the Variance of the Return on Market
        covarReturnStock = sumProduct/n
        beta = covarReturnStock / varReturnMarket
        alpha = meanReturnStock - meanReturnMarket*beta
        st.header("Result")
        # st.write(f"* **the Variance of the Return on Market**: {varReturnMarket:.3f}")
        # st.write(f"* the Variance of the Return on Stock has been calculated as {covarReturnStock:.3f}")
      
        firstCol, secCol = st.columns(2)
        firstCol.metric(label="**Beta variable**", value=beta.round(3))
        secCol.metric(label="**Alpha variable**", value=alpha.round(3))
