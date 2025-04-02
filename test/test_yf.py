import yfinance as yf
import pandas as pd

# futures_data = yf.download("PL=F", start="2017-01-01", end="2021-12-31")

ticker = "0002.HK"
df= yf.download(ticker, period='30y', actions=True)

df.to_csv(f"{ticker}.csv")


# dat = yf.Ticker("0002.HK")
# print(dat.info)

# print(dat.calendar)
# print(dat.analyst_price_targets)
# print(dat.quarterly_income_stmt)
# print(dat.history(period='20y'))
# print(dat.option_chain(dat.options[0]).calls)


