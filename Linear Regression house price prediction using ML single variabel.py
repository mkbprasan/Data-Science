#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[41]:


ln=pd.read_csv("C:\\Users\\prasann\\Desktop\\DS\\ML Proj\DataSets_ML Projects\\Linear Regression Proj\\LN.csv")


# In[42]:


ln


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('house price prediction')
plt.xlabel('area in sqfeet')
plt.ylabel('price')
plt.scatter(ln.area,ln.price,color='red',marker='+')


# In[48]:


reg=linear_model.LinearRegression()


# In[ ]:





# In[50]:


reg.fit(ln[['area']],ln.price)


# In[51]:


reg.predict([[3300]]) 


# In[53]:


df1=pd.read_excel("C:\\Users\\prasann\\Desktop\\DS\\ML Proj\DataSets_ML Projects\\Linear Regression Proj\\values.xlsx")


# In[54]:


df1


# In[56]:


p=reg.predict(df1)


# In[58]:


df1['price']=p


# In[61]:


df1.to_excel('prediction.xlsx')


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area',fontsize=20) 
plt.ylabel('price',fontsize=20)
plt.scatter(df1.area,df1.price,color='red',marker='+')
plt.plot(df1['area'],reg.predict(df1[['area']]),color='blue')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




