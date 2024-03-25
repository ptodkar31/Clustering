# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:56:23 2023

@author: Priyanka
"""


"""
Perform clustering analysis on the telecom data set. 
The data is a mixture of both categorical and numerical data. 
It consists of the number of customers who churn out. 
Derive insights and get possible information on factors that may 
affect the churn decision.
Refer to Telco_customer_churn.xlsx dataset
 
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("C:\Data Set\Telco_customer_churn.xlsx")
df.head()
df.columns

# There are many unwanted columns of data 
# hence drop that columns
df=df.drop(['Customer ID','Offer'],axis=1)

# There are some nominal type data columns 
# so use dummies for that

df.shape
#(7043, 28)
df1=pd.get_dummies(df)

df1.shape
#(7043, 48)
df1.columns

df1=df1.drop(['Referred a Friend_Yes','Phone Service_Yes','Multiple Lines_Yes',
          'Internet Service_Yes','Online Security_Yes','Online Backup_Yes','Device Protection Plan_Yes'
          ,'Premium Tech Support_Yes','Streaming TV_Yes','Streaming Movies_Yes','Streaming Music_Yes','Unlimited Data_Yes','Paperless Billing_Yes'],axis=1,inplace=True)

df1.shape
# Now we get 30,12
df1=df1.rename(columns={'Referred a Friend_No':'Referred a Friend',
'Phone Service_No':'Phone Service', 'Multiple Lines_No':'Multiple Lines', 'Internet Service_No':'Internet Service',
'Online Security_No':'Online Security', 'Online Backup_No':'Online Backup', 'Device Protection Plan_No':'Device Protection Plan',
'Premium Tech Support_No':'Premium Tech Support', 'Streaming TV_No':'Streaming TV', 'Streaming Movies_No':'Streaming Movies',
'Streaming Music_No':'Streaming Music', 'Unlimited Data_No':'Unlimited Data', 'Paperless Billing_No':'Paperless Billing'})

# There is scale diffrence between among the columns hence normalize it
# whenever there is mixed data apply normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# drop the Quarter_Q3 showing Nan value after norm function
df1=df1.drop(['Quarter_Q3'],axis=1)
# count is Nan
df_norm=norm_fun(df1.iloc[:,1:])

b=df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")

# sch.dendrogram(z)
# leaf-rotation for rotate 
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
# dendrongram()
# applying agglomerative clustering choosing 3 as clustrers
# from dendrongram
# whatever has been displayed in dendrogram is not clustering
# It is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)
# apply labels to clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

# Assign this series to df Dataframe as column and name the column
df1['clust']=cluster_labels
# we want to restore the column 7 to 0 th position
df1.shape

# now check the df dataframe
df1.iloc[:,2:].groupby(df1.clust).mean()
# from the output cluster 2 has got highest Top10
# lowest accept ratio , best faculty ratio and highest expenses
# highest graduate 

df.to_csv("Teclo.csv",encoding="utf-8")
import os
os.getcwd()
