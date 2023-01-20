#!/usr/bin/env python
# coding: utf-8

# T20 World Cup 2022 Analysis using Python

# Every sports event generates a lot of data which we can use to analyze the performance of players, teams, and many highlights of the game. As the ICC Men’s T20 world cup has finished ,it has generated a lot of data we can use to summarize the event. So, if you want to learn how to analyze a sports event like the t20 world cup, . This will take you through the task of T20 World Cup 2022 analysis using Python.

# The dataset contains data about all the matches from the super 12 stage to the final of the ICC Men’s T20 World Cup 2022. Below are all the features in the dataset:
# 
# venue: The venue where the match was played
# team1: the team that batted first
# team2: the team that batted second
# stage: stage of the match (super 12, semi-final, or final)
# toss winner: the team that won the toss
# toss decision: the decision of the captain after winning the toss
# first innings score: runs scored in the first innings
# first innings wickets: the number of wickets lost in the first innings
# second innings score: runs scored in the second innings
# second innings wickets: the number of wickets lost in the second innings
# winner: the team that won the match
# won by: how the team won the match (wickets or runs)
# player of the match: the player of the match
# top scorer: the player who scored highest in the match
# highest score: the highest runs scored in the match by the player
# best bowler: the player who took the most wickets in the match
# best bowling figure: the number of wickets taken and runs given by the best bowler in the match
# You can use this dataset for analyzing and summarizing the ICC men’s t20 world cup 2022.

# T20 World Cup 2022 Analysis using Python
# The dataset we are using for the T20 World Cup 2022 analysis is collected manually.

# Now let’s start this task by importing the necessary Python libraries and the dataset:

# In[10]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("C:\\Users\\prasann\Desktop\\DS\\ML Proj\\DataSets_ML Projects\\T20 World Cup 2022 Dataset\\t20-world-cup-22.csv")
print(data.head())


# Now let’s look at the number of matches won by each team in the world cup:

# In[11]:


figure = px.bar(data, 
                x=data["winner"],
                title="Number of Matches Won by teams in t20 World Cup 2022")
figure.show()


# As England won the t20 world cup 2022, England won five matches. And Both Pakistan and India won 4 matches.

# Now let’s have a look at the number of matches won by batting first or second in the t20 world cup 2022:

# In[12]:


won_by = data["won by"].value_counts()
label = won_by.index
counts = won_by.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Number of Matches Won By Runs Or Wickets')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


# So in the t20 world cup 2022, 16 matches were won by batting first, and 13 matches were won by chasing. Now, let’s have a look at the toss decisions by teams in the world cup:

# In[13]:


toss = data["toss decision"].value_counts()
label = toss.index
counts = toss.values
colors = ['skyblue','yellow']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Toss Decisions in t20 World Cup 2022')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


# So in 17 matches, the teams decided to bat first, and in 13 matches, the teams chose to field first. Now let’s have a look at the top scorers in the t20 world cup 2022:

# In[14]:


figure = px.bar(data, 
                x=data["top scorer"], 
                y = data["highest score"], 
                color = data["highest score"],
                title="Top Scorers in t20 World Cup 2022")
figure.show()


# So, Virat Kohli scored the highest in 3 matches. Undoubtedly, he was the best batsman in the t20 world cup 2022. Now let’s have a look at the number of player of the match awards in the world cup:

# In[15]:


figure = px.bar(data, 
                x = data["player of the match"], 
                title="Player of the Match Awards in t20 World Cup 2022")
figure.show()


# Virat Kohli, Sam Curran, Taskin Ahmed, Suryakumar Yadav, and Shadab Khan got the player of the match in 2 matches. No player got the player of the match award in more than two matches.
# 
# Now let’s have a look at the bowlers with the best bowling figures at the end of the matches:

# In[16]:


figure = px.bar(data, 
                x=data["best bowler"],
                title="Best Bowlers in t20 World Cup 2022")
figure.show()


# Sam Curran was the only best bowler in 3 matches. Undoubtedly, he deserved to be the player of the tournament. Now let’s compare the runs scored in the first innings and second innings in every stadium of the t20 world cup 2022:

# In[17]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=data["venue"],
    y=data["first innings score"],
    name='First Innings Runs',
    marker_color='blue'
))
fig.add_trace(go.Bar(
    x=data["venue"],
    y=data["second innings score"],
    name='Second Innings Runs',
    marker_color='red'
))
fig.update_layout(barmode='group', 
                  xaxis_tickangle=-45, 
                  title="Best Stadiums to Bat First or Chase")
fig.show()


# So SCG was the only stadium in the world cup that was best for batting first. Other stadiums didn’t make much difference while batting first or chasing.

# Now let’s compare the number of wickets lost in the first innings and second innings in every stadium of the t20 world cup 2022:

# In[19]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=data["venue"],
    y=data["first innings wickets"],
    name='First Innings Wickets',
    marker_color='blue'
))
fig.add_trace(go.Bar(
    x=data["venue"],
    y=data["second innings wickets"],
    name='Second Innings Wickets',
    marker_color='red'
))
fig.update_layout(barmode='group', 
                  xaxis_tickangle=-45, 
                  title="Best Statiums to Bowl First or Defend")
fig.show()


# SCG was the best stadium to bowl while defending the target. While the Optus Stadium was the best stadium to bowl first.

# Summary
# So some highlights of the t20 world cup 2022 we found from our analysis are:
# 
# England won the most number of matches
# Virat Kohli scored highest in the most number of matches
# Sam Curran was the best bowler in the most number of matches
# More teams won by batting first
# More teams decided to bat first
# SCG was the best stadium to bat first
# SCG was the best stadium to defend the target in the World Cup
# The Optus Stadium was the best stadium to bowl first
# I hope you liked this article on the t20 world cup 2022 analysis using Python. Feel free to ask valuable questions in the comments section below.
# 
# 

# In[ ]:




