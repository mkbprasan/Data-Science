#!/usr/bin/env python
# coding: utf-8

# Twitter Stock Market Analysis using Python
Twitter is one of the popular social media applications where people share what they feel in a limited number of words. Twitter is popular but not in the stock market. As Twitter is being delisted from the New York stock exchange, let’s analyze the complete timeline of Twitter in the stock market. In this article, I will take you through the task of Twitter Stock Market analysis using Python.
# Twitter started its journey in the stock market in 2013. So to analyze the complete timeline of Twitter in the stock market, we need the stock price data of Twitter from 2013 to 202
Now let’s start with the task of Twitter Stock Market Analysis by importing the necessary Python libraries and the dataset:
# Twitter was founded in 2006 and got listed on the stock exchange in 2013. Since the founding of Twitter, 2022 has been an event to remember Twitter. As Elon Musk took over Twitter, it will be delisted from the New York Exchange. As 2022 was so eventful for Twitter, analyze the complete timeline of Twitter in the Stock Market from 2013 to 2022.
# 
# Here is the Twitter stock market data you can use. Below are all the features in the data:
# 
# Date: Date of the record
# Open: Opening price
# High: Highest price recorded
# Low: Lowest price recorded
# Close: Closing price
# Adj Close: Adjusted Closing price
# Volume: The total number of shares traded
# Analyze the complete timeline of Twitter in the Stock Market from 2013 to 2022.

# Now let’s start with the task of Twitter Stock Market Analysis by importing the necessary Python libraries and the dataset:

# In[1]:


import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("C:\\Users\\prasann\\Desktop\\DS\\ML Proj\\DataSets_ML Projects\\Twitter Stock Market Analysis\\TWTR.csv")
print(data.head())

The dataset contains data about:

Date
The opening Price of the day
The highest price of the day
The lowest price of the day
The closing price of the day
The adjusted closing price of the day
The total number of shares traded in the day (volume)
# Let’s have a look at the column insights:

# In[2]:


print(data.info())


# The Date column is an object in this dataset. We will convert it into a datetime data type later. Now, let’s have a look at whether this dataset contains any null values or not:

# In[3]:


print(data.isnull().sum())


# In[4]:


data.shape


# In[5]:


data = data.dropna()


# Now let’s have a look at the stock prices of Twitter over the years:

# In[6]:


figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = "Twitter Stock Prices Over the Years", 
                     xaxis_rangeslider_visible=False)
figure.show()


# So since the introduction of Twitter in the stock market, it was only profitable at the beginning and 2021. Let’s visualize a bar chart to analyze the stock prices of Twitter in detail:

# In[7]:


figure = px.bar(data, 
                x = "Date", 
                y= "Close", 
                color="Close")
figure.update_xaxes(rangeslider_visible=True)
figure.show()


# The above graph shows the stock prices of Twitter over the years. You can use the range slider below to zoom into a particular time period. Look at the image as an example

# Using the range slider, we can see that the first quarter of 2021 was the best time period for Twitter over the years in the stock market. We can also assign buttons to control time periods. Let’s add buttons to analyze the stock prices of Twitter in different time periods:

# In[9]:


figure = px.bar(data, x = "Date", y= "Close", color="Close")
figure.update_xaxes(rangeslider_visible=True)
figure.update_layout(title = "Twitter Stock Prices Over the Years", 
                     xaxis_rangeslider_visible=False)
figure.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
figure.show()


# Now let’s have a look at the complete timeline of Twitter in the stock market:

# In[10]:


data["Date"] = pd.to_datetime(data["Date"], 
                              format = '%Y-%m-%d')
data['Year'] = data['Date'].dt.year
data["Month"] = data["Date"].dt.month
fig = px.line(data, 
              x="Month", 
              y="Close", 
              color='Year', 
              title="Complete Timeline of Twitter")
fig.show()


# So since the introduction of Twitter in the stock market, 2014 went well for Twitter in the first four years. 2016 and 2017 were the worst for Twitter in the stock market. Its stock prices went up in 2018, 2019, and 2020. And then came 2021, the best year for Twitter in the stock market. Twitter reached its highest-ever stock price in the year 2021. But the stock prices of Twitter went down again in 2022.
Summary
So this is how you can analyze the complete timeline of Twitter in the stock market from 2013 to 2022. Twitter is a popular social media application and is still getting more popular after Elon Musk took over Twitter. But it never was among the best-performing companies in the stock market. I hope you liked this article on Twitter Stock Market Analysis using Python. Feel free to ask valuable questions in the comments section below.