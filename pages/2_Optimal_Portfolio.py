# Import Libraries
from utils import functions
import streamlit as st
import datetime
import yahoo_fin.stock_info as si
import numpy as np
from pandas import DataFrame, concat
from itertools import combinations
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Optimal Portfolio",
    page_icon="💰",
)

header = st.container()
params = st.container()
userInput = st.container()
stocksSelected = st.container()
dataset = st.container()
lineChart = st.container()
results = st.container()
scatterChart = st.container()
relationMatrixes = st.container()
simulation = st.container()
optimalPortfolio = st.container()

with header:
    st.title("Optimal Portfolio")

with params:
    marketOptions = ["S&P 500 Index", "NASDAQ-100 Index", "Russell 1000 Index"]


with userInput:
    st.markdown("Please Provide Information Needed")
    # Requires user selection from the stock list
    marketSelection = st.selectbox("Select the market:", options=marketOptions)
    stockDict = functions.stocks(marketSelection)
    choosenStockList = st.multiselect(
        "Select the companies whose shares you are interested in:", options=stockDict.keys())
    riskPerception = st.selectbox("Your risk perception", options=[
                                  "Risk neutral", "Risk lover", "Risk averse"])
    # Defines variables
    choosenTickerList = []
    for stock in choosenStockList:
        choosenTickerList.append(stockDict[stock])
    firstCol, secCol = st.columns(2)
    # Assigns one year ago as default beginning date
    defaultBegDate = datetime.datetime(datetime.datetime.now(
    ).year-1, datetime.datetime.now().month, datetime.datetime.now().day)
    # Calender components for user's date entry
    startDate = firstCol.date_input(
        "Pick a start date ", value=defaultBegDate, max_value=datetime.datetime.now())
    endDate = secCol.date_input(
        "Pick an end date", min_value=startDate, max_value=datetime.datetime.now())
    if len(choosenStockList) > 1:
        st.write("You have choosen the stocks below.")
    firstCol, secCol = st.columns(2)
    for i in range(len(choosenStockList)):
        firstCol._text_input(
            f"Stock {i+1}", choosenStockList[i], disabled=True)
        secCol._text_input(
            f"Ticker {i+1}", choosenTickerList[i], disabled=True)
    st.write("\n")

with dataset:
    historicalData = DataFrame()
    riskFreeData = DataFrame()

    try:
        for stockOption in choosenTickerList:
            historicalData[stockOption] = si.get_data(
                stockOption, start_date=startDate, end_date=endDate)['adjclose']
    except AssertionError:
        print("Returns for these dates not available")
        pass

    # Market Data
    historicalData["SP500"] = si.get_data(
        "^GSPC", start_date=startDate, end_date=endDate)['adjclose']

    riskFreeData["3-months T-Bill"] = si.get_data(
        "^IRX", start_date=startDate, end_date=endDate)['adjclose']
    rf = riskFreeData["3-months T-Bill"].mean()
    returnRates = historicalData.pct_change()*100
    # Deletes first row
    returnRates = returnRates.iloc[1:, :]

    # annualReturnList["AAPL"] or annualReturnList[0] to call variable
    annualReturnList = returnRates.mean()*250
    stdList = np.std(returnRates)
    meanList = returnRates.mean()

with lineChart:
    st.header("Line Charts")
    st.line_chart((historicalData / historicalData.iloc[0]*100))

with results:
    mainTable = DataFrame()
    df = concat([stdList, annualReturnList], axis=1)
    df.columns = ['Standard Deviation', 'Annual Return']
    df['Sharpe Ratio'] = (df['Annual Return']-rf)/df['Standard Deviation']
    df = df.sort_values("Annual Return", ascending=False).round(4)
    # Sorts assets in purpose of preventing miscalculations
    assets = []
    for i in df.index:
        if i == 'SP500':
            pass
        else:
            assets.append(i)

    st.header("Annual Return & Standard Deviation")
    st.table(df)

with scatterChart:
    fig, ax = plt.subplots()
    ax.set_title('Annual Return Rate & Standard Deviation Rate')
    plt.xlabel('Standard Deviation Rate')
    plt.ylabel('Annual Return Rate')
    plt.scatter(df['Standard Deviation'], df['Annual Return'],
                label="Standard Deviation & Annual Return", c=df['Sharpe Ratio'], cmap="plasma")
    cbar = plt.colorbar()
    cbar.set_label('Sharpe Ratio')
    for i, txt in enumerate(df.index):
        ax.annotate(txt, (df['Standard Deviation'][i], df['Annual Return'][i]))
    st.pyplot(fig)


with relationMatrixes:
    st.header("Covariance Matrix")
    covMatrix = returnRates.cov().round(4)
    st.table(covMatrix.style.apply(functions.hightlight_diag, axis=None))
    st.markdown(
        "Please note that the diagonal values are the variances of the assets respectively.")
    st.write("\n")

    st.header("Coefficient Correlation Matrix")
    corrMatrix = returnRates.corr().round(4)
    st.table(corrMatrix)
    st.write("\n")

with simulation:
    # MONTE CARLO SIMULATION
    iterList = list(combinations(choosenTickerList, 2))
    assetList1 = []
    assetList2 = []
    portfolioWeightList1 = []
    portfolioWeightList2 = []
    portfolioReturnList1 = []
    portfolioReturnList2 = []
    portfolioStdErrorList1 = []
    portfolioStdErrorList2 = []
    rhoList = []
    for i in iterList:
        # Creates a dataframe of portfolios which contains weights of assets and assets' ticker names
        for x in range(1000):
            weights = np.random.random(2)
            weights /= np.sum(weights)
            assetList1.append(i[0])
            assetList2.append(i[1])
            portfolioWeightList1.append(weights[0])
            portfolioWeightList2.append(weights[1])
            portfolioReturnList1.append(df.loc[i[0]]['Annual Return'])
            portfolioReturnList2.append(df.loc[i[1]]['Annual Return'])
            portfolioStdErrorList1.append(df.loc[i[0]]['Standard Deviation'])
            portfolioStdErrorList2.append(df.loc[i[1]]['Standard Deviation'])
            rhoList.append(corrMatrix[i[0]][i[1]])
        portfolios = DataFrame({'Asset 1': assetList1, 'Asset 2': assetList2, 'Weight 1': portfolioWeightList1, 'Weight 2': portfolioWeightList2, 'Return 1': portfolioReturnList1,
                                  'Return 2': portfolioReturnList2, 'Volatility 1': portfolioStdErrorList1, 'Volatility 2': portfolioStdErrorList2, 'Correlation': rhoList})

    try:
        w1 = portfolios['Weight 1']
        w2 = portfolios['Weight 2']
        r1 = portfolios['Return 1']
        r2 = portfolios['Return 2']
        s1 = portfolios['Volatility 1']
        s2 = portfolios['Volatility 2']
        rho = portfolios['Correlation']
        portfolios['Annual Return'] = w1 * r1 + w2 * r2
        portfolios['Volatility'] = np.sqrt(
            abs(np.square(w1)*np.square(s1) + np.square(w2)*np.square(s2) + 2*w1*w2*rho))
        portfolios['Theta'] = (
            portfolios['Annual Return'] - rf)/portfolios['Volatility']

        fig, ax = plt.subplots()
        ax.set_title('Efficient Frontiers Between Assets')
        # portfolios.plot(x='Volatility', y='Annual Return', kind='scatter', figsize=(18,12))
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.scatter(portfolios['Volatility'],
                    portfolios['Annual Return'], s=0.2)
        for i, txt in enumerate(df.index):
            if txt == 'SP500':
                pass
            else:
                ax.annotate(txt, (df['Standard Deviation']
                            [i], df['Annual Return'][i]))
        st.pyplot(fig)
    except NameError:
        st.write(
            "**You haven't choose stocks to see any result. Please choose stocks!**")
        st.write("")
    with optimalPortfolio:
        weights = []
        volatilities = []
        returns = []

        for i in iterList:
            r1 = df.loc[i[0]]['Annual Return']
            r2 = df.loc[i[1]]['Annual Return']
            s1 = df.loc[i[0]]['Standard Deviation']
            s2 = df.loc[i[1]]['Standard Deviation']
            if r1 > r2:
                w2 = ((r2 - rf)*np.square(s1) - (r1 - rf)*covMatrix[i[0]][i[1]]) / ((r2 - rf)*np.square(
                    s1) + (r1 - rf)*np.square(s2) - (r1 + r2 - 2*rf)*covMatrix[i[0]][i[1]])
                w1 = 1 - w2
            else:
                w1 = ((r1 - rf)*np.square(s2) - (r2 - rf)*covMatrix[i[0]][i[1]]) / ((r1 - rf)*np.square(
                    s2) + (r2 - rf)*np.square(s1) - (r1 + r2 - 2*rf)*covMatrix[i[0]][i[1]])
                w1 = 1 - w2
            weights.append([w1, w2])
            returns.append(w1 * r1 + w2 * r2)
            volatilities.append(np.sqrt(abs(np.square(
                w1)*np.square(s1) + np.square(w2)*np.square(s2) + 2*w1*w2*corrMatrix[i[0]][i[1]])))

        # Creates a dataframe for optimal portfolios
        optimal = DataFrame(weights, columns=["w1", "w2"], index=iterList)
        optimal['expected return'] = returns
        optimal['volatility'] = volatilities

        st.subheader("Optimal Risky Portfolios based on your selection")
        #optimal = optimal.sort_values("expected return", ascending=False)
        st.table(optimal)

        for i in optimal.index:
            try:
                for j in optimal.index:
                    if i == j:
                        pass
                    elif optimal.volatility[i] < optimal.volatility[j]:
                        if optimal['expected return'][i] > optimal['expected return'][j]:
                            optimal = optimal.drop(labels=j, axis=0)
                        else:
                            continue
            except KeyError:
                continue

        st.subheader("Optimal Risky Portfolios after the intelligent elimination")
        st.table(optimal)

        fig, ax = plt.subplots()
        ax.set_title('Efficient Frontiers Between Assets')
        for i in optimal.index:
            plt.scatter(portfolios.loc[portfolios['Asset 1'] ==i[0]].loc[portfolios['Asset 2'] ==i[1]].Volatility, portfolios.loc[portfolios['Asset 1'] ==i[0]].loc[portfolios['Asset 2'] ==i[1]]['Annual Return'], color='tab:orange', s=0.1)
        plt.scatter(0, rf, marker="x")
        plt.scatter(optimal.volatility,
                    optimal['expected return'], color='tab:red', marker="*")
        for i, txt in enumerate(optimal.index):
            ax.annotate(txt[0]+", "+txt[1], (optimal['volatility']
                        [i], optimal['expected return'][i]))
        # portfolios.plot(x='Volatility', y='Annual Return', kind='scatter', figsize=(18,12))
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        st.pyplot(fig)

        # Draws Capital Allocation Line
        fig, ax = plt.subplots()
        for i in optimal.index:
            x_values = [0, optimal.volatility[i]]
            y_values = [rf, optimal['expected return'][i]]

            plt.plot(x_values, y_values, 'bo', linestyle="-")
            ax.plot()
            ax.annotate(i[0]+", "+i[1], (optimal.volatility[i],
                        optimal['expected return'][i]))
            ax.annotate("3 months T-bill", (0, rf))
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        st.pyplot(fig)

        if riskPerception == "Risk lover":
            A = 2
        elif riskPerception == "Risk neutral":
            A = 3
        else:       # Risk Averse
            A = 4

        portfolioWeights = []
        index = []
        for i in optimal.index:
            portfolioWeight = (optimal['expected return'][i] - rf ) / ( A * (optimal.volatility[i]**2))
            row = [optimal['w1'][i]*portfolioWeight/100,optimal['w2'][i]*portfolioWeight/100, 1-portfolioWeight/100]
            index.append((i[0], i[1], "3 months T-bill"))
            portfolioWeights.append(row)
        st.subheader("Optimal Portfolios")
        optimalData = DataFrame(np.array(portfolioWeights), columns = ['Weight 1', 'Weight 2', 'Weight 3'], index = index)
        st.table(optimalData)