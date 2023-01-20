#!/usr/bin/env python
# coding: utf-8

# Credit Score Classification
# There are three credit scores that banks and credit card companies use to label their customers:
# 
# Good
# Standard
# Poor
# A person with a good credit score will get loans from any bank and financial institution. For the task of Credit Score Classification, we need a labelled dataset with credit scores.
# 
# I found an ideal dataset for this task labelled according to the credit history of credit card customers. You can download the dataset here.
# 
# In the section below, I will take you through the task of credit score classification with Machine Learning using Python

# Credit Score Classification using Python
# Let’s start the task of credit score classification by importing the necessary Python libraries and the dataset:

# Credit Score Classification: Case Study
# The credit score of a person determines the creditworthiness of the person. It helps financial companies determine if you can repay the loan or credit you are applying for.
# 
# Here is a dataset based on the credit score classification submitted by Rohan Paris on Kaggle. Below are all the features in the dataset:
# 
# ID: Unique ID of the record
# Customer_ID: Unique ID of the customer
# Month: Month of the year
# Name: The name of the person
# Age: The age of the person
# SSN: Social Security Number of the person
# Occupation: The occupation of the person
# Annual_Income: The Annual Income of the person
# Monthly_Inhand_Salary: Monthly in-hand salary of the person
# Num_Bank_Accounts: The number of bank accounts of the person
# Num_Credit_Card: Number of credit cards the person is having
# Interest_Rate: The interest rate on the credit card of the person
# Num_of_Loan: The number of loans taken by the person from the bank
# Type_of_Loan: The types of loans taken by the person from the bank
# Delay_from_due_date: The average number of days delayed by the person from the date of payment
# Num_of_Delayed_Payment: Number of payments delayed by the person
# Changed_Credit_Card: The percentage change in the credit card limit of the person
# Num_Credit_Inquiries: The number of credit card inquiries by the person
# Credit_Mix: Classification of Credit Mix of the customer
# Outstanding_Debt: The outstanding balance of the person
# Credit_Utilization_Ratio: The credit utilization ratio of the credit card of the customer
# Credit_History_Age: The age of the credit history of the person
# Payment_of_Min_Amount: Yes if the person paid the minimum amount to be paid only, otherwise no.
# Total_EMI_per_month: The total EMI per month of the person
# Amount_invested_monthly: The monthly amount invested by the person
# Payment_Behaviour: The payment behaviour of the person
# Monthly_Balance: The monthly balance left in the account of the person
# Credit_Score: The credit score of the person
# The Credit_Score column is the target variable in this problem. You are required to find relationships based on how banks classify credit scores and train a model to classify the credit score of a person.

# The Credit_Score column is the target variable in this problem. You are required to find relationships based on how banks classify credit scores and train a model to classify the credit score of a person.

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("C:\\Users\\prasann\\Desktop\\DS\\ML Proj\\DataSets_ML Projects\\Credit Score Classification\\train.csv")
print(data.head())


# Let’s have a look at the information about the columns in the dataset:

# In[2]:


print(data.info())


# Before moving forward, let’s have a look if the dataset has any null values or not:

# In[3]:


print(data.isnull().sum())


# The dataset doesn’t have any null values. As this dataset is labelled, let’s have a look at the Credit_Score column values:

# In[4]:


data["Credit_Score"].value_counts()


# Data Exploration
# The dataset has many features that can train a Machine Learning model for credit score classification. Let’s explore all the features one by one.
# 
# I will start by exploring the occupation feature to know if the occupation of the person affects credit scores:

# In[5]:


fig = px.box(data, 
             x="Occupation",  
             color="Credit_Score", 
             title="Credit Scores Based on Occupation", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.show()


# There’s not much difference in the credit scores of all occupations mentioned in the data. Now let’s explore whether the Annual Income of the person impacts your credit scores or not:

# In[6]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Annual_Income", 
             color="Credit_Score",
             title="Credit Scores Based on Annual Income", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# According to the above visualization, the more you earn annually, the better your credit score is. Now let’s explore whether the monthly in-hand salary impacts credit scores or not:

# In[7]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Inhand_Salary", 
             color="Credit_Score",
             title="Credit Scores Based on Monthly Inhand Salary", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# Like annual income, the more monthly in-hand salary you earn, the better your credit score will become. Now let’s see if having more bank accounts impacts credit scores or not:

# In[10]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Num_Bank_Accounts", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Bank Accounts", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# Maintaining more than five accounts is not good for having a good credit score. A person should have 2 – 3 bank accounts only. So having more bank accounts doesn’t positively impact credit scores. Now let’s see the impact on credit scores based on the number of credit cards you have

# In[14]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Num_Credit_Card", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Credit cards", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# Just like the number of bank accounts, having more credit cards will not positively impact your credit scores. Having 3 – 5 credit cards is good for your credit score. Now let’s see the impact on credit scores based on how much average interest you pay on loans and EMIs:

# In[15]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Interest_Rate", 
             color="Credit_Score",
             title="Credit Scores Based on the Average Interest rates", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# If the average interest rate is 4 – 11%, the credit score is good. Having an average interest rate of more than 15% is bad for your credit scores. Now let’s see how many loans you can take at a time for a good credit score

# In[16]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Num_of_Loan", 
             color="Credit_Score", 
             title="Credit Scores Based on Number of Loans Taken by the Person",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# To have a good credit score, you should not take more than 1 – 3 loans at a time. Having more than three loans at a time will negatively impact your credit scores. Now let’s see if delaying payments on the due date impacts your credit scores or not:

# In[17]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Delay_from_due_date", 
             color="Credit_Score",
             title="Credit Scores Based on Average Number of Days Delayed for Credit card Payments", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# So you can delay your credit card payment 5 – 14 days from the due date. Delaying your payments for more than 17 days from the due date will impact your credit scores negatively. Now let’s have a look at if frequently delaying payments will impact credit scores or not

# In[27]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Num_of_Delayed_Payment", 
            color="Credit_Score", 
             title="Credit Scores Based on Number of Delayed Payments",
           color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# So delaying 4 – 12 payments from the due date will not affect your credit scores. But delaying more than 12 payments from the due date will affect your credit scores negatively. Now let’s see if having more debt will affect credit scores or not:

# In[28]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Outstanding_Debt", 
             color="Credit_Score", 
             title="Credit Scores Based on Outstanding Debt",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# An outstanding debt of $380 – $1150 will not affect your credit scores. But always having a debt of more than $1338 will affect your credit scores negatively. Now let’s see if having a high credit utilization ratio will affect credit scores or not

# In[29]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_Utilization_Ratio", 
             color="Credit_Score",
             title="Credit Scores Based on Credit Utilization Ratio", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# Credit utilization ratio means your total debt divided by your total available credit. According to the above figure, your credit utilization ratio doesn’t affect your credit scores. Now let’s see how the credit history age of a person affects credit scores:

# In[30]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_History_Age", 
             color="Credit_Score", 
             title="Credit Scores Based on Credit History Age",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# So, having a long credit history results in better credit scores. Now let’s see how many EMIs you can have in a month for a good credit score:

# In[31]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Total_EMI_per_month", 
             color="Credit_Score", 
             title="Credit Scores Based on Total Number of EMIs per Month",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# The number of EMIs you are paying in a month doesn’t affect much on credit scores. Now let’s see if your monthly investments affect your credit scores or not:

# In[32]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Amount_invested_monthly", 
             color="Credit_Score", 
             title="Credit Scores Based on Amount Invested Monthly",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# The amount of money you invest monthly doesn’t affect your credit scores a lot. Now let’s see if having a low amount at the end of the month affects credit scores or not:

# In[33]:


fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Balance", 
             color="Credit_Score", 
             title="Credit Scores Based on Monthly Balance Left",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# So, having a high monthly balance in your account at the end of the month is good for your credit scores. A monthly balance of less than $250 is bad for credit scores.

# Credit Score Classification Model
# One more important feature (Credit Mix) in the dataset is valuable for determining credit scores. The credit mix feature tells about the types of credits and loans you have taken.
# 
# As the Credit_Mix column is categorical, I will transform it into a numerical feature so that we can use it to train a Machine Learning model for the task of credit score classification:

# In[34]:


data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, 
                               "Good": 2, 
                               "Bad": 0})


# Now I will split the data into features and labels by selecting the features we found important for our model:

# In[35]:


from sklearn.model_selection import train_test_split
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data[["Credit_Score"]])


# Now, let’s split the data into training and test sets and proceed further by training a credit score classification model:

# In[38]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)


# Now, let’s make predictions from our model by giving inputs to our model according to the features we used to train the model:

# In[ ]:


print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))


# In[ ]:





# So this is how you can use Machine Learning for the task of Credit Score Classification using Python

# In[ ]:




