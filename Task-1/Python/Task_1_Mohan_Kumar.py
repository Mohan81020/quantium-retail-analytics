#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


customer = pd.read_csv("E:\Quanties_Data_Analyst_Project\QVI_purchase_behaviour.csv")
transaction = pd.read_csv("E:\Quanties_Data_Analyst_Project\QVI_transaction_data.csv")


# In[3]:


customer.head()


# In[4]:


customer.tail()


# In[5]:


transaction.head()


# In[6]:


transaction.tail()


# In[7]:


customer.columns


# In[8]:


transaction.columns


# In[9]:


customer.info()


# In[10]:


transaction.info()


# In[11]:


customer.describe()


# In[12]:


transaction.describe()


# In[13]:


#checking Null values


# In[14]:


customer.isnull().sum()


# In[15]:


transaction.isnull().sum()


# In[16]:


# checking Duplicated Values 


# In[17]:


customer.duplicated().sum()


# In[18]:


transaction.duplicated().sum()


# In[19]:


transaction.drop_duplicates(inplace= True)


# In[20]:


transaction.duplicated().sum()


# In[21]:


customer.nunique()


# In[22]:


transaction.nunique()


# In[23]:


customer['LIFESTAGE'].unique()


# In[24]:


customer['PREMIUM_CUSTOMER'].unique()


# In[25]:


customer['LYLTY_CARD_NBR'].duplicated().sum()


# In[26]:


# Checking Date Range and Gap in Date Columns


# In[27]:


Min_Date = transaction['DATE'].min()
Max_Date = transaction['DATE'].max()
full_range = pd.date_range(Max_Date,Min_Date)
missing_date = full_range.difference(transaction['DATE'].unique())
print(missing_date)


# In[28]:


# Checking outlier 


# In[29]:


plt.boxplot(transaction['PROD_QTY'])
plt.show()


# In[30]:


transaction['PROD_QTY'].unique()


# In[31]:


# Filltering all records with 200 prod qty 


# In[32]:


transaction[transaction['PROD_QTY'] > 10].value_counts()


# In[33]:


transaction = transaction[transaction['PROD_QTY'] < 10]


# In[34]:


plt.boxplot(transaction['PROD_QTY'])
plt.show()


# In[35]:


plt.boxplot(transaction['TOT_SALES'])
plt.show()


# In[36]:


transaction['PROD_NAME'].head(10)


# In[37]:


# Feature Engineering


# In[38]:


transaction["PACK_SIZE"] = transaction["PROD_NAME"].str.extract('(\d+)').astype(int)


# In[39]:


transaction['BRAND'] = transaction['PROD_NAME'].str.split().str[0]


# In[40]:


transaction.head()


# In[41]:


transaction["PACK_SIZE"]


# In[42]:


transaction["BRAND"].value_counts().sort_index()


# In[43]:


transaction[transaction["BRAND"] == "Old"]["PROD_NAME"].unique()


# In[44]:


transaction.loc[
    transaction["PROD_NAME"].str.startswith("Natural Chip Compny"),
    "BRAND"
] = "Natural Chip Compny"

transaction.loc[
    transaction["PROD_NAME"].str.startswith("Old El Paso"),
    "BRAND"
] = "Old El Paso"

transaction.loc[
    transaction["PROD_NAME"].str.startswith("Grain Waves"),
    "BRAND"
] = "Grain Waves"

transaction.loc[
    transaction["PROD_NAME"].str.startswith("Red Rock Deli"),
    "BRAND"
] = "Red Rock Deli"


# In[45]:


transaction.head()


# In[46]:


# Merging both datasets


# In[47]:


df_final = transaction.merge(customer,on = "LYLTY_CARD_NBR", how = 'left')


# In[48]:


df_final.info()


# In[49]:


df_final.isnull().sum()


# In[50]:


df_final.duplicated().sum()


# In[51]:


# Exploratory Data Analysis 


# In[52]:


df_final.columns


# In[53]:


# Total Sales
df_final['TOT_SALES'].sum()


# In[54]:


# Total Transactions

df_final['TXN_ID'].nunique()


# In[55]:


# total Customers 
df_final['LYLTY_CARD_NBR'].nunique()


# In[56]:


# Total Qty Sold 
df_final['PROD_QTY'].sum()


# In[57]:


df_final.rename(columns=str.lower, inplace=True)


# In[58]:


df_final.head()


# In[59]:


# Created the segment column
df_final['segment'] = df_final['lifestage'] + " - " + df_final['premium_customer']


# In[60]:


df_final['segment'].head()


# In[61]:


# Total sales by segment

sales_by_segment = df_final.groupby(df_final['segment'])['tot_sales'].sum().reset_index()
sales_by_segment = sales_by_segment.sort_values('tot_sales',ascending = False)
sales_by_segment


# In[62]:


# Number of unique customers by segment

customers_by_segment  = df_final.groupby(['segment'])['lylty_card_nbr'].nunique().reset_index()
customers_by_segment.columns =  ['segment','num_cust']


# In[63]:


customers_by_segment


# In[64]:


# Average spend per customer

segment_summary = sales_by_segment.merge(customers_by_segment, on = 'segment')
segment_summary['Avg_spend_per_cust'] = segment_summary['tot_sales']/ segment_summary['num_cust']


# In[65]:


segment_summary.head()


# In[66]:


#  Average transactions per customer (frequency)

taxn_by_segment = df_final.groupby('segment')['txn_id'].nunique().reset_index()
taxn_by_segment.columns = ['segment','txn_num']


# In[67]:


segment_summary = segment_summary.merge(taxn_by_segment,on = 'segment')
segment_summary['Avg_trans_per_cust'] = segment_summary['txn_num']/segment_summary['num_cust']


# In[68]:


segment_summary.head()


# In[69]:


#  Average units per transaction (basket size)

segment_summary['Avg_units_per_trans'] = df_final.groupby(['segment'])['prod_qty'].sum().values/segment_summary['txn_num'].values


# In[70]:


segment_summary.head()


# In[71]:


# Average price per unit
total_price = df_final.groupby('segment')['tot_sales'].sum().values
total_qty = df_final.groupby('segment')['prod_qty'].sum().values
segment_summary['Avg_price_per_unit'] = total_price/total_qty


# In[72]:


segment_summary.head()


# In[73]:


df_final.columns


# In[74]:


# Preferred pack size by segment


# In[75]:


pack_seg = df_final.groupby(['segment','pack_size'])['prod_qty'].sum().reset_index()

# total units sold of each pack size across ALL customers (baseline)

pack_total = df_final.groupby(['pack_size'])['prod_qty'].sum().reset_index()
pack_total.columns = ['pack_size','total_prod_qty']
total_qty = df_final['prod_qty'].sum()
pack_total['pack_share_overall'] = pack_total['total_prod_qty']/total_qty
#pack_total.head()

seg_total_unit = df_final.groupby('segment')['prod_qty'].sum().reset_index()
seg_total_unit.columns =['segment','total_unit_segment']
#seg_total_unit

pack_seg =pack_seg.merge(seg_total_unit,on = 'segment').merge(pack_total,on='pack_size')
# pack_seg.head()

pack_seg['pack_share_segment'] = pack_seg['prod_qty']/pack_seg['total_unit_segment']
pack_seg['affinity_index'] = pack_seg['pack_share_segment'] / pack_seg['pack_share_overall'] * 100
pack_seg.head()


# In[76]:


# Preferred brand by segment

brand_seg = df_final.groupby(['segment','brand'])['prod_qty'].sum().reset_index()

# total units sold of each brand across ALL customers (baseline)

brand_total = df_final.groupby('brand')['prod_qty'].sum().reset_index()
brand_total.columns = ['brand','total_qty_sold']
#brand_total.head()
total_qty = df_final['prod_qty'].sum()

brand_total['brand_share_overall'] = brand_total['total_qty_sold']/total_qty

#brand_total.head()
#seg_total_unit.head()

brand_seg = brand_seg.merge(seg_total_unit,on='segment').merge(brand_total,on='brand')
#brand_seg.head()

brand_seg['brand_share_segment'] =brand_seg['prod_qty']/brand_seg['total_unit_segment']
brand_seg['affinity_index'] = brand_seg['brand_share_segment']/brand_seg['brand_share_overall'] * 100

brand_seg.head()


# In[77]:


# Visualization

# Total sales by segment
plt.figure(figsize=(10,6))
plt.barh(segment_summary['segment'], segment_summary['tot_sales'])
plt.xlabel('Total Sales')
plt.title('Total Chip Sales by Customer Segment')
plt.tight_layout()
plt.show()



# In[78]:


# Avg spend per customer by segment
plt.figure(figsize=(10,6))
plt.barh(segment_summary['segment'], segment_summary['Avg_trans_per_cust'])
plt.xlabel('Average Spend per Customer')
plt.title('Average Spend per Customer by Segment')
plt.tight_layout()
plt.show()


# In[89]:


df_final['lylty_card_nbr']


# In[ ]:




