#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# 从Excel文件读取数据
file_path = r'C:\Users\86135\Desktop\附件2.xlsx'
try:
    df = pd.read_excel(file_path, header=None, names=['Date', 'Time', 'ItemID', 'Quantity'])
    
    # 查看数据的列名和前几行数据
    print("数据列名:", df.columns)
    print("数据预览:")
    print(df.head())

except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")

# 确保列名正确
if 'Date' in df.columns and 'ItemID' in df.columns:
    # 将数据按日期分组
    transactions = df.groupby('Date')['ItemID'].apply(list).tolist()
    
    # 转换为适合 FP-growth 的格式
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 查看编码后的数据（可选）
    print("\n编码后的数据:")
    print(df_encoded.head())


# In[ ]:





# In[ ]:




