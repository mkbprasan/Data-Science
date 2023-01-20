#!/usr/bin/env python
# coding: utf-8

# Weather Forecasting using Python

# In Data Science, weather forecasting is an application of Time Series Forecasting where we use time-series data and algorithms to make forecasts for a given time. If you want to learn how to forecast the weather using your Data Science skills, this article is for you. In this article, I will take you through the task of weather forecasting using Python

# Weather Forecasting
# Weather forecasting is the task of forecasting weather conditions for a given location and time. With the use of weather data and algorithms, it is possible to predict weather conditions for the next n number of days.
# 
# For forecasting weather using Python, we need a dataset containing historical weather data based on a particular location. I found a dataset on Kaggle based on the Daily weather data of New Delhi. We can use this dataset for the task of weather forecasting.

# In the section below, you will learn how we can analyze and forecast the weather using Python.

# Analyzing Weather Data using Python

# Now let’s start this task by importing the necessary Python libraries and the dataset we need:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("C:\\Users\\prasann\\Desktop\\DS\\ML Proj\\DataSets_ML Projects\\Weather Forecasting\\DailyDelhiClimateTrain.csv")
print(data.head())


# Let’s have a look at the descriptive statistics of this data before moving forward:

# In[2]:


print(data.describe())


# Now let’s have a look at the information about all the columns in the dataset:

# In[3]:


print(data.info())


# The date column in this dataset is not having a datetime data type. We will change it when required. Let’s have a look at the mean temperature in Delhi over the years:

# In[4]:


figure = px.line(data, x="date", 
                 y="meantemp", 
                 title='Mean Temperature in Delhi Over the Years')
figure.show()


# Now let’s have a look at the humidity in Delhi over the years:

# In[5]:


figure = px.line(data, x="date", 
                 y="humidity", 
                 title='Humidity in Delhi Over the Years')
figure.show()


# Now let’s have a look at the wind speed in Delhi over the years:

# In[6]:


figure = px.line(data, x="date", 
                 y="wind_speed", 
                 title='Wind Speed in Delhi Over the Years')
figure.show()


# Till 2015, the wind speed was higher during monsoons (August & September) and retreating monsoons (December & January). After 2015, there were no anomalies in wind speed during monsoons. Now let’s have a look at the relationship between temperature and humidity:

# In[7]:


figure = px.scatter(data_frame = data, x="humidity",
                    y="meantemp", size="meantemp", 
                    trendline="ols", 
                    title = "Relationship Between Temperature and Humidity")
figure.show()


# There’s a negative correlation between temperature and humidity in Delhi. It means higher temperature results in low humidity and lower temperature results in high humidity.

# # Analyzing Temperature Change

# Now let’s analyze the temperature change in Delhi over the years. For this task, I will first convert the data type of the date column into datetime. Then I will add two new columns in the dataset for year and month values.
# 
# 

# Here’s how we can change the data type and extract year and month data from the date column:

# In[8]:


data["date"] = pd.to_datetime(data["date"], format = '%Y-%m-%d')
data['year'] = data['date'].dt.year
data["month"] = data["date"].dt.month
print(data.head())


# Now let’s have a look at the temperature change in Delhi over the years:

# In[9]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x='month', y='meantemp', hue='year')
plt.show()


# Although 2017 was not the hottest year in the summer, we can see a rise in the average temperature of Delhi every year.

# Forecasting Weather using Python
# Now let’s move to the task of weather forecasting. I will be using the Facebook prophet model for this task. The Facebook prophet model is one of the best techniques for time series forecasting. If you have never used this model before, you can install it on your system by using the command mentioned below in your command prompt or terminal

# In[10]:


pip install prophet


# The prophet model accepts time data named as “ds”, and labels as “y”. So let’s convert the data into this format:

# In[11]:


forecast_data = data.rename(columns = {"date": "ds", 
                                       "meantemp": "y"})
print(forecast_data)


# Now below is how we can use the Facebook prophet model for weather forecasting using Python:

# In[12]:


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions)


# So this is how you can analyze and forecast the weather using Python.
# 
# Summary
# Weather forecasting is the task of forecasting weather conditions for a given location and time. With the use of weather data and algorithms, it is possible to predict weather conditions for the next n number of days. I hope you liked this article on Weather Analysis and Forecasting using Python. Feel free to ask valuable questions in the comments section below.

# In[ ]:




