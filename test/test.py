#####-----------------------------------------------#####
##### Download historical stock data from yfinance using JSON
#####-----------------------------------------------#####
# By Meltec, 2024
# Written on Jupyter 7.0.8, Python 3.12.4
# For educational purposes and personal use only. USE AT YOUR OWN RISK AND DISCRETION
# If it's helpful, kindly acknowledge Meltec and the website (https://meltec22.wixsite.com/finlitsg)
# Donations welcome on the site. Every little bit helps! :)
import urllib.request
import json
import numpy as np
import time
import math
###-----------------------------------------------
### Parameter settings
###-----------------------------------------------
# Number of days of data to pull (backwards from today). Max is 1year (365 days)
days = 90
# end date (today) in epoch time, truncated to GMT 0000:00:00
tend = int((time.time())//(60*60*24) *(60*60*24))
# start date in epoch time, truncated to GMT 0000:00:00
tstart = tend - (days*24*60*60)
# List of symbols to pull from yahoo finance
symbols = ["ES3.SI","D05.SI","VWRA.L","BTC-USD"]
###-----------------------------------------------
### Define function to pull data from yfinance in JSON format
###-----------------------------------------------
def read_json(symbol):
    try:
        myurl = "https://query1.finance.yahoo.com/v8/finance/chart/%s?metrics=adjclose?&interval=1d&range=1y" % symbol
        myrequest = urllib.request.Request(url=myurl,headers = {'User-agent': 'Chrome/134.0.0.0'})
        with urllib.request.urlopen(myrequest) as body:
            rawdata = json.load(body)
    except:
            raise Exception("URL Error")
    return rawdata['chart']['result'][0]
###-----------------------------------------------
### Merge data for all symbols, dates without price data uses previous day's prices
###-----------------------------------------------
# Initialize array
alldata = np.zeros([1+np.size(symbols), days+1])
# 0th column is date (tstart to tend), nth column is symbol.
alldata[0,:]= [int(x) for x in range(tstart,tend+1,(24*60*60))]
# Loop through symbols
for n in range(np.size(symbols)):
    # Call read_json function
    rawdata = read_json(symbols[n])
    # Initialize array for cleaning up pulled JSON data
    cleandata = np.zeros([2,np.size(rawdata['timestamp'])])
    # Pulled data from yfinance, column 0: date
    cleandata[0,:] = [x//(60*60*24) *(60*60*24) for x in rawdata['timestamp']]
    # Pulled data from yfinance, column 1: adjusted closing price
    cleandata[1,:] = rawdata['indicators']['adjclose'][0]['adjclose']
    # Initialize row index for pulled data
    i = 0
    # Loop through our desired date range
    for d in range(days+1):
        # While our start date is after the date in pulled data,
        # and we are not at the last entry, go to next row in pulled data.
        while (alldata[0,d] > cleandata[0,i]) and (i < (np.size(rawdata['timestamp'])-1)):
            i+=1
        # If our start date matches one of the dates in pulled data, save the pulled adjclose to symbol's column.
        # If its not a number, take last pulled value instead
        if alldata[0,d] == cleandata[0,i]:
            j = i
            while math.isnan(cleandata[1,j]):
                j-=1
            alldata[n+1,d] = cleandata[1,j]
            # Then move on to next row in pulled data, if we are not at last entry.
            if i < (np.size(rawdata['timestamp'])-1):
                i+=1
        # Else if next available pulled data is after our desired date
        # (e.g. We want saturday's data but next available data after friday is monday)
        # Take previous day's pulled adjclose. Keep row index constant (e.g. until our monday = pulled monday)
        # If i=0, skip and leave that day's adjclose as 0
        elif (alldata[0,d] < cleandata[0,i]) and (i>0):
            alldata[n+1,d] = cleandata[1,i-1]
# alldata[] now has columns containing dates and the symbols.
# Rows are our date range and adjclose data for that day (or previous trading day's adjclose)
###-----------------------------------------------
### Post-processing and export
###-----------------------------------------------
# Create header row
symbols.insert(0,"Date")
# Convert our epoch dates to human readable dates YYYYMMDD (Option 1)
alldata[0,:]= [str(time.strftime('%Y%m%d', time.gmtime(x))) for x in alldata[0,:]]
# Convert our epoch dates to Excel dates (Option 2)
# alldata[0,:]= [x//86400 +25569 for x in alldata[0,:]]
# Export to HistoricalAdjClose.csv. Make sure file does not exist, or is closed before running. Else will get permission error
np.savetxt("HistoricalAdjClose.csv",np.transpose(alldata),header=str(symbols),fmt='%.4f',delimiter=",",comments='')