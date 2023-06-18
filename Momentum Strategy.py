#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


# In[2]:


Index_Composition = pd.read_csv("C:/Users/dipan/OneDrive/Desktop/ResNet/INDEXCOMPOSITION[25].csv", 
                                usecols = list(range(0,102)),index_col = 0,header = None)
Index_Composition.index = pd.to_datetime(Index_Composition.index, format='%d-%m-%Y')
Index_Composition.sort_index(axis=0,inplace=True)


# In[3]:


Index_Returns = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/ResNet/INDEXDATA[52].csv',index_col = 0)
Index_Returns.index = pd.to_datetime(Index_Returns.index, format='%Y-%m-%d')


# In[4]:


Stock_Returns = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/ResNet/STOCKDATA[3].csv',index_col = 0)
Stock_Returns.index = pd.to_datetime(Stock_Returns.index, format='%Y-%m-%d')


# In[5]:


Index_Returns = Index_Returns[Index_Returns.index.isin(Stock_Returns.index)]
Index_Composition = Index_Composition[Index_Composition.index.isin(Stock_Returns.index)]


# In[6]:


Stock_Returns_Monthly = Stock_Returns.copy()
Stock_Returns_Monthly.fillna(0, inplace=True)
Stock_Returns_Monthly +=1
Stock_Returns_Monthly['Month'] = Stock_Returns_Monthly.index.month
Stock_Returns_Monthly['Year'] = Stock_Returns_Monthly.index.year
Stock_Returns_Monthly = Stock_Returns_Monthly.groupby(['Year','Month']).prod()
Stock_Returns_Monthly = Stock_Returns_Monthly - 1
Stock_Returns_Monthly.reset_index(inplace=True)


# In[7]:


Index_Returns_Monthly = Index_Returns.copy()
Index_Returns_Monthly +=1
Index_Returns_Monthly['Month'] = Index_Returns_Monthly.index.month
Index_Returns_Monthly['Year'] = Index_Returns_Monthly.index.year
Index_Returns_Monthly = Index_Returns_Monthly.groupby(['Year','Month']).prod() - 1
Index_Returns_Monthly.reset_index(inplace=True)


# In[8]:


Stock_Beta = Stock_Returns.apply(lambda x: np.cov(np.array(Index_Returns[x.notnull()]['NIFTY50']),np.array(x[x.notnull()]))[1,0]/np.var(np.array(Index_Returns[x.notnull()]['NIFTY50'])),
                    axis=0)

Stock_SD = Stock_Returns.apply(lambda x: np.std(x),axis=0)


# In[9]:


First_Dates = pd.DataFrame({'Year':Stock_Returns_Monthly['Year'],
                           'Month':Stock_Returns_Monthly['Month'],
                           'First_Date':Stock_Returns.groupby(Stock_Returns.index.to_period("m")).head(1).index})


# In[10]:


Stock_Returns_12mo = pd.concat([Stock_Returns_Monthly[['Year','Month']],((Stock_Returns_Monthly.drop(['Year','Month'],axis=1)+1).rolling(12, min_periods=12, axis=0).apply(np.prod)-1)],axis=1)


# In[11]:


Stock_Returns_ID =  pd.concat([Stock_Returns_Monthly[['Year','Month']],Stock_Returns_Monthly.drop(['Year','Month'],axis=1).rolling(12,axis=0).apply(lambda x: np.sign(np.prod(x+1)-1)*(sum(x<0)-sum(x>0)))],axis=1)


# In[12]:


Stock_Returns_12mo_rank = pd.concat([Stock_Returns_Monthly[['Year','Month']],Stock_Returns_12mo.drop(['Year','Month'],axis=1).rank(axis=1,method='first')],axis=1)


# In[13]:


Stock_Returns_ID_rank = pd.concat([Stock_Returns_Monthly[['Year','Month']],Stock_Returns_ID.drop(['Year','Month'],axis=1).rank(axis=1,method='first')],axis=1)


# In[14]:


Stock_Returns_next_month = pd.concat([Stock_Returns_Monthly[['Year','Month']],Stock_Returns_Monthly.drop(['Year','Month'],axis=1).shift(-1)],axis=1)


# In[15]:


# Stock_Returns_Total_Rank = pd.concat([Stock_Returns_Monthly[['Year','Month']],175 - Stock_Returns_ID_rank.drop(['Year','Month'],axis=1) + Stock_Returns_12mo_rank.drop(['Year','Month'],axis=1)],axis


# In[16]:


# with pd.ExcelWriter('C:\\Users\\dipan\\OneDrive\\Desktop\\Momentum Raw Data.xlsx') as writer:
#     Stock_Returns_Monthly.to_excel(writer, sheet_name='monthly returns')
#     Stock_Returns_12mo.to_excel(writer, sheet_name='12 mo returns')
#     Stock_Returns_12mo_rank.to_excel(writer, sheet_name='12 mo returns rank')
#     Stock_Returns_ID.to_excel(writer, sheet_name='12 mo ID')
#     Stock_Returns_ID_rank.to_excel(writer, sheet_name='12 mo ID rank')
#     Stock_Returns_next_month.to_excel(writer, sheet_name='next month returns')
#     Stock_Returns_Total_Rank.to_excel(writer, sheet_name='total rank')    


# In[17]:


n_largest = 10

min_returns_order = np.argsort(Stock_Returns_12mo_rank.drop(['Year','Month'],axis=1).values, axis=1)[:,:n_largest]
min_returns = pd.DataFrame(Stock_Returns_12mo_rank.columns[min_returns_order+2],
             columns=['bottom{}'.format(i) for i in range(1, n_largest+1)],
             index=Stock_Returns_12mo_rank.index)

max_returns_order = np.argsort(-Stock_Returns_12mo_rank.drop(['Year','Month'],axis=1).values, axis=1)[:,:n_largest]
max_returns = pd.DataFrame(Stock_Returns_12mo_rank.columns[max_returns_order+2],
             columns=['top{}'.format(i) for i in range(1, n_largest+1)],
             index=Stock_Returns_12mo_rank.index)


Long_Short_Returns_Df = pd.DataFrame(columns = ['Year','Month','Long 1','Long 2','Long 3',
                                             'Short 1','Short 2','Short 3',
                                            'Long Return 1','Long Return 2','Long Return 3',
                                            'Short Return 1','Short Return 2','Short Return 3'])
for i in range(11,Stock_Returns_12mo.shape[0]):
    long_list = Stock_Returns_ID[Stock_Returns_ID.columns.intersection(max_returns.iloc[i])].iloc[i]
    long_select = []
    for j,x in enumerate(max_returns.iloc[i]):
        if long_list[x]<=-10:
            long_select.append(max_returns.iloc[i,j])
        if len(long_select)==3:
            break

    if len(long_select)<3:
        for j,x in enumerate(max_returns.iloc[i]):
            if long_list[x]==-9:
                long_select.append(max_returns.iloc[i,j])
            if len(long_select)==3:
                break

    if len(long_select)<3:
        for j,x in enumerate(max_returns.iloc[i]):
            if long_list[x]==-8:
                long_select.append(max_returns.iloc[i,j])
            if len(long_select)==3:
                break

    if len(long_select)<3:
        for j,x in enumerate(max_returns.iloc[i]):
            if long_list[x]==-7:
                long_select.append(max_returns.iloc[i,j])
            if len(long_select)==3:
                break

    if len(long_select)<3:
        for j,x in enumerate(max_returns.iloc[i]):
            if max_returns.iloc[i,j] in long_select:
                continue
            else:
                long_select.append(max_returns.iloc[i,j])
            if len(long_select)==3:
                break

    short_list = Stock_Returns_ID[Stock_Returns_ID.columns.intersection(min_returns.iloc[i])].iloc[i]            
    short_select = []
    for j,x in enumerate(min_returns.iloc[i]):
        if short_list[x]<=-6:
            short_select.append(min_returns.iloc[i,j])
        if len(short_select)==3:
            break

    if len(short_select)<3:
        for j,x in enumerate(min_returns.iloc[i]):
            if short_list[x]==-5:
                short_select.append(min_returns.iloc[i,j])
            if len(short_select)==3:
                break

    if len(short_select)<3:
        for j,x in enumerate(min_returns.iloc[i]):
            if short_list[x]==-4:
                short_select.append(min_returns.iloc[i,j])
            if len(short_select)==3:
                break

    if len(short_select)<3:
        for j,x in enumerate(min_returns.iloc[i]):
            if short_list[x]==-3:
                short_select.append(min_returns.iloc[i,j])
            if len(short_select)==3:
                break

    if len(short_select)<3:
        for j,x in enumerate(min_returns.iloc[i]):
            if min_returns.iloc[i,j] in short_select:
                continue
            else:
                short_select.append(min_returns.iloc[i,j])
            if len(short_select)==3:
                break


    Long_Short_Returns_Df.loc[len(Long_Short_Returns_Df.index)] = list(Stock_Returns_next_month[['Year','Month']].iloc[i]) + list(Stock_Returns_next_month[long_select+short_select].iloc[i].index.values) + list(Stock_Returns_next_month[long_select+short_select].iloc[i].values)

Long_Short_Returns_Df = Long_Short_Returns_Df.iloc[:-1]


# In[ ]:





# In[18]:


Dates = First_Dates['First_Date'].iloc[11:143]

Benchmark = Index_Returns_Monthly['NIFTY50'].iloc[11:143]
Benchmark_Cum = (1+Benchmark).cumprod()


# In[19]:


#Cross_sectional only strategies
#Strategy 1 - Long Only Equal Weight
S1 = Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3']].apply(lambda x: np.mean(x),axis=1)
S1_Cum = (1+S1).cumprod()

plt.plot(Dates,S1_Cum, label = 'Long Only Equal Weight')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[20]:


#Strategy 1.1 - Long Only Equal Weight Only Positive Returns
S1_1 = []
S1_1_stock_count = []
for i in range(Long_Short_Returns_Df.shape[0]):
    Returns_12mo = Stock_Returns_12mo[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]].iloc[i+11]
    flag = [1 if i>0 else 0 for i in Returns_12mo]
    S1_1_stock_count.append(sum(np.array(flag)))
    w = np.array(flag)/sum(np.array(flag))
    S1_1.append(np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3']].iloc[i],w))

#Turns out all stocks have positive returns


# In[21]:


#Strategy 2 - Long Only Risk Parity
S2 = []
for i in range(Long_Short_Returns_Df.shape[0]):
    w = 1/Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]/sum(1/Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]])
    S2.append(np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3']].iloc[i],w))
S2_Cum = (np.array(S2)+1).cumprod()
plt.plot(Dates,S2_Cum, label = 'Long Only Risk Parity')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[22]:


import scipy.optimize as optim
def objective_fn_market_neutral(x,beta):
    x_temp = [x[0],x[0],x[0],x[1],x[1],x[1]]
    return np.dot(x_temp,beta)**2


# In[23]:


#Strategy 3 - Market neutral - Equal wt
S3 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    B_long = Stock_Beta[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]
    B_short = Stock_Beta[Long_Short_Returns_Df[['Short 1','Short 2','Short 3']].iloc[i]]
    beta = pd.concat([B_long,B_short])
    x0 = [.5,-.3]
    bnds = ((0,1),(-1,0))
    cons =  ({'type':'eq', 'fun':lambda x: np.sum(x)*3-1})
    res= optim.minimize(objective_fn_market_neutral,x0,method='SLSQP',
                        constraints = cons, bounds=bnds,args = (beta), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][0],res['x'][0],res['x'][1],res['x'][1],res['x'][1]]
    w_list.append(w)
    S3.append(np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3',
                                            'Short Return 1','Short Return 2','Short Return 3']].iloc[i],w))

S3_Cum = (np.array(S3)+1).cumprod()
plt.plot(Dates,S3_Cum, label = 'Market Neutral Equal Weight')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()

# print(w_list)


# In[24]:


def objective_fn_market_neutral_long_1_short_1(x,beta):
    x_temp = [x[0],x[1]]
    return np.dot(x_temp,beta)**2

#Strategy 3.1 - Market neutral - Equal wt
S3_1 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    B_long = Stock_Beta[Long_Short_Returns_Df[['Long 1']].iloc[i]]
    B_short = Stock_Beta[Long_Short_Returns_Df[['Short 1']].iloc[i]]
    beta = pd.concat([B_long,B_short])
    x0 = [1,-1]
    bnds = ((0,2),(-2,0))
    cons =  ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    res= optim.minimize(objective_fn_market_neutral_long_1_short_1,x0,method='SLSQP',
                        constraints = cons,bounds=bnds,args = (beta), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][1]]
    w_list.append(w)
    S3_1.append(np.dot(Long_Short_Returns_Df[['Long Return 1','Short Return 1']].iloc[i],w))

S3_1_Cum = (np.array(S3_1)+1).cumprod()
plt.plot(Dates,S3_1_Cum, label = 'Market Neutral Equal Weight - 1 Long 1 Short')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()
# print(w_list)


# In[25]:


def objective_fn_risk_budgeting(x,sigma_sq,sigma_budget_sq):
    return (np.dot(x,sigma_sq) - sigma_budget_sq)**2


# In[26]:


#Strategy 4 - Risk budgeting
S4 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    SD_long = Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]
    SD_short = Stock_SD[Long_Short_Returns_Df[['Short 1','Short 2','Short 3']].iloc[i]]
    SD = pd.concat([SD_long,SD_short],ignore_index = True)
    SD_sq = SD**2
    x0 = [.5,.5,.5,-.3,-.3,-.3]
    bnds = ((0,1),(0,1),(0,1),(-1,0),(-1,0),(-1,0))
    sigma_budget = 0.1
    cons =  ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    res= optim.minimize(objective_fn_risk_budgeting,x0,method='SLSQP',
                        constraints = cons,bounds=bnds,args = (SD_sq,sigma_budget**2), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][1],res['x'][2],res['x'][3],res['x'][4],res['x'][5]]
    w_list.append(w)
    S4.append(np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3',
                                            'Short Return 1','Short Return 2','Short Return 3']].iloc[i],w))

S4_Cum = (np.array(S4)+1).cumprod()
plt.plot(Dates,S4_Cum, label = 'Risk Budgeting')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()


# In[27]:


#Strategy 5 - Long Only Risk budgeting
S5 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    SD = Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]
#     SD_short = Stock_SD[Long_Short_Returns_Df[['Short 1','Short 2','Short 3']].iloc[i]]
#     SD = pd.concat([B_long,B_short])
    SD_sq = SD**2
    x0 = [.5,.5,0]
    bnds = ((0,1),(0,1),(0,1))
    sigma = 0.15
    res= optim.minimize(objective_fn_risk_budgeting,x0,method='SLSQP',
                        bounds=bnds,args = (SD_sq,sigma**2), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][1],res['x'][2]]
    w_list.append(w)
    S5.append(np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3']].iloc[i],w))

S5_Cum = (np.array(S5)+1).cumprod()
plt.plot(Dates,S5_Cum, label = 'Long Only Risk Budgeting')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()


# In[28]:


plt.plot(Dates,S1_Cum, label = 'Long Only Equal Weight')
plt.plot(Dates,S2_Cum, label = 'Long Only Risk Parity')
plt.plot(Dates,S3_Cum, label = 'Market Neutral Equal Weight')
plt.plot(Dates,S3_1_Cum, label = 'Market Neutral Equal Weight - 1 Long 1 Short')
plt.plot(Dates,S4_Cum, label = 'Risk Budgeting')
plt.plot(Dates,S5_Cum, label = 'Long Only Risk Budgeting')
plt.plot(Dates,Benchmark_Cum, label = 'Nifty 50')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[29]:


#Cross-sectional and time series strategies
Index = (Index_Returns+1).cumprod()*100
Index.columns = ['Index']
Index['SMA_20'] = Index['Index'].rolling(20).mean()
Index['SMA_50'] = Index['Index'].rolling(50).mean()
Index['SMA_100'] = Index['Index'].rolling(50).mean()
Index['SMA_200'] = Index['Index'].rolling(200).mean()

plt.figure(figsize=(10,6))
plt.plot(Index.index,Index['Index'], label = 'Nifty 50')
# plt.plot(Index.index,Index['SMA_20'], label = 'SMA 20')
# plt.plot(Index.index,Index['SMA_50'], label = 'SMA 50')
plt.plot(Index.index,Index['SMA_100'], label = 'SMA 100')
plt.plot(Index.index,Index['SMA_200'], label = 'SMA 200')
plt.legend()
plt.show()


# In[30]:


Index_Returns_Mon = pd.concat([Index_Returns_Monthly, Index.iloc[Index.index.isin(First_Dates['First_Date'])].reset_index(drop=True)], axis =1)


# In[31]:


Index_Returns_Mon['Flag_20'] = [1 if Index_Returns_Mon['Index'].iloc[i]>Index_Returns_Mon['SMA_20'].iloc[i] else 0 for i in range(Index_Returns_Mon.shape[0])]
Index_Returns_Mon['Flag_50'] = [1 if Index_Returns_Mon['Index'].iloc[i]>Index_Returns_Mon['SMA_50'].iloc[i] else 0 for i in range(Index_Returns_Mon.shape[0])]
Index_Returns_Mon['Flag_100'] = [1 if Index_Returns_Mon['Index'].iloc[i]>Index_Returns_Mon['SMA_100'].iloc[i] else 0 for i in range(Index_Returns_Mon.shape[0])]
Index_Returns_Mon['Flag_200'] = [1 if Index_Returns_Mon['Index'].iloc[i]>Index_Returns_Mon['SMA_200'].iloc[i] else 0 for i in range(Index_Returns_Mon.shape[0])]


# In[32]:


print(Index_Returns_Mon.groupby('Flag_20')['NIFTY50'].mean())
print(Index_Returns_Mon.groupby('Flag_50')['NIFTY50'].mean())
print(Index_Returns_Mon.groupby('Flag_100')['NIFTY50'].mean())
print(Index_Returns_Mon.groupby('Flag_200')['NIFTY50'].mean())


# In[33]:


print(Index_Returns_Mon.groupby('Flag_20')['NIFTY50'].std())
print(Index_Returns_Mon.groupby('Flag_50')['NIFTY50'].std())
print(Index_Returns_Mon.groupby('Flag_100')['NIFTY50'].std())
print(Index_Returns_Mon.groupby('Flag_200')['NIFTY50'].std())


# In[34]:


#Cash Rate
rf = 0.04


# In[35]:


Long_Short_Returns_Df['Flag_20'] = Index_Returns_Mon['Flag_20'].iloc[11:142].reset_index(drop=True)
Long_Short_Returns_Df['Flag_50'] = Index_Returns_Mon['Flag_50'].iloc[11:142].reset_index(drop=True)
Long_Short_Returns_Df['Flag_100'] = Index_Returns_Mon['Flag_100'].iloc[11:142].reset_index(drop=True)
Long_Short_Returns_Df['Flag_200'] = Index_Returns_Mon['Flag_200'].iloc[11:142].reset_index(drop=True)


# In[36]:


#Strategy 6 - Long Only Equal Weight or Cash
S6 = Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3','Flag_200']].apply(lambda x: (x[0]+x[1]+x[2])/3 if x[3]>0 else rf,axis=1)
S6_Cum = (1+S6).cumprod()

plt.plot(Dates,S6_Cum, label = 'Long Only Equal Weight or Cash')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[37]:


#Strategy 7 - Long Only Risk Parity or Cash
S7 = []
for i in range(Long_Short_Returns_Df.shape[0]):
    w = 1/Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]/sum(1/Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]])
    ret = np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3']].iloc[i],w)
    ret = ret if Long_Short_Returns_Df['Flag_200'].iloc[i]>0 else rf
    S7.append(ret)
S7_Cum = (np.array(S7)+1).cumprod()
plt.plot(Dates,S7_Cum, label = 'Long Only Risk Parity or Cash')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[38]:


#Strategy 8 - Market neutral - Equal wt or Cash
S8 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    B_long = Stock_Beta[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]
    B_short = Stock_Beta[Long_Short_Returns_Df[['Short 1','Short 2','Short 3']].iloc[i]]
    beta = pd.concat([B_long,B_short])
    x0 = [.5,-.3]
    bnds = ((0,1),(-1,0))
    cons =  ({'type':'eq', 'fun':lambda x: np.sum(x)*3-1})
    res= optim.minimize(objective_fn_market_neutral,x0,method='SLSQP',
                        constraints = cons, bounds=bnds,args = (beta), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][0],res['x'][0],res['x'][1],res['x'][1],res['x'][1]]
    w_list.append(w)
    ret = np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3',
                                            'Short Return 1','Short Return 2','Short Return 3']].iloc[i],w)
    ret = ret if Long_Short_Returns_Df['Flag_200'].iloc[i]>0 else rf
    S8.append(ret)

S8_Cum = (np.array(S8)+1).cumprod()
plt.plot(Dates,S8_Cum, label = 'Market Neutral Equal Weight or Cash')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[39]:


#Strategy 9 - Risk budgeting or Cash
S9 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    SD_long = Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]
    SD_short = Stock_SD[Long_Short_Returns_Df[['Short 1','Short 2','Short 3']].iloc[i]]
    SD = pd.concat([SD_long,SD_short],ignore_index = True)
    SD_sq = SD**2
    x0 = [.5,.5,.5,-.3,-.3,-.3]
    bnds = ((0,1),(0,1),(0,1),(-1,0),(-1,0),(-1,0))
    sigma_budget = 0.1
    cons =  ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    res= optim.minimize(objective_fn_risk_budgeting,x0,method='SLSQP',
                        constraints = cons,bounds=bnds,args = (SD_sq,sigma_budget**2), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][1],res['x'][2],res['x'][3],res['x'][4],res['x'][5]]
    w_list.append(w)
    ret = np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3',
                                            'Short Return 1','Short Return 2','Short Return 3']].iloc[i],w)
    ret = ret if Long_Short_Returns_Df['Flag_200'].iloc[i]>0 else rf
    S9.append(ret)

S9_Cum = (np.array(S9)+1).cumprod()
plt.plot(Dates,S9_Cum, label = 'Risk Budgeting or Cash')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[40]:


#Strategy 10 - Long Only Risk budgeting or Cash
S10 = []
w_list = []
for i in range(Long_Short_Returns_Df.shape[0]):
    SD = Stock_SD[Long_Short_Returns_Df[['Long 1','Long 2','Long 3']].iloc[i]]
#     SD_short = Stock_SD[Long_Short_Returns_Df[['Short 1','Short 2','Short 3']].iloc[i]]
#     SD = pd.concat([B_long,B_short])
    SD_sq = SD**2
    x0 = [.5,.5,0]
    bnds = ((0,1),(0,1),(0,1))
    sigma = 0.15
    res= optim.minimize(objective_fn_risk_budgeting,x0,method='SLSQP',
                        bounds=bnds,args = (SD_sq,sigma**2), options = {'maxiter':1000})
    w = [res['x'][0],res['x'][1],res['x'][2]]
    w_list.append(w)
    ret = np.dot(Long_Short_Returns_Df[['Long Return 1','Long Return 2','Long Return 3']].iloc[i],w)
    ret = ret if Long_Short_Returns_Df['Flag_200'].iloc[i]>0 else rf
    S10.append(ret)

S10_Cum = (np.array(S10)+1).cumprod()
plt.plot(Dates,S10_Cum, label = 'Long Only Risk Budgeting or Cash')
plt.plot(Dates,Benchmark_Cum, label = 'Benchmark')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[41]:


plt.plot(Dates,S6_Cum, label = 'Long Only Equal Weight or Cash')
plt.plot(Dates,S7_Cum, label = 'Long Only Risk Parity or Cash')
plt.plot(Dates,S8_Cum, label = 'Market Neutral Equal Weight or Cash')
plt.plot(Dates,S9_Cum, label = 'Risk Budgeting or Cash')
plt.plot(Dates,S10_Cum, label = 'Long Only Risk Budgeting or Cash')
plt.plot(Dates,Benchmark_Cum, label = 'Nifty 50')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[42]:


plt.plot(Dates,S1_Cum, label = 'Long Only Equal Weight')
plt.plot(Dates,S6_Cum, label = 'Long Only Equal Weight or Cash')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[43]:


plt.plot(Dates,S2_Cum, label = 'Long Only Risk Parity')
plt.plot(Dates,S7_Cum, label = 'Long Only Equal Weight or Cash')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[44]:


plt.plot(Dates,S3_Cum, label = 'Market Neutral Equal Weight')
plt.plot(Dates,S8_Cum, label = 'Market Neutral Equal Weight or Cash')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[45]:


plt.plot(Dates,S4_Cum, label = 'Risk Budgeting')
plt.plot(Dates,S9_Cum, label = 'Risk Budgeting or Cash')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[46]:


plt.plot(Dates,S5_Cum, label = 'Long Only Risk Budgeting')
plt.plot(Dates,S10_Cum, label = 'Long Only Risk Budgeting or Cash')
plt.title("Rs 1 invested")
plt.legend()
plt.show()


# In[47]:


Strategies = {'Benchmark':Benchmark.reset_index(drop=True),
     'Long Only Equal Weight': S1,
     'Long Only Risk Parity': S2,
     'Market Neutral Equal Weight': S3,
     'Risk Budgeting': S4,
     'Long Only Risk Budgeting': S5,
     'Long Only Equal Weight or Cash': S6,
     'Long Only Risk Parity or Cash': S7,
     'Market Neutral Equal Weight or Cash': S8,
     'Risk Budgeting or Cash': S9,
     'Long Only Risk Budgeting or Cash': S10}

Strategies = pd.concat([pd.Series(v, name=k) for k, v in Strategies.items()], axis=1)
Strategies.index = First_Dates['First_Date'].iloc[11:143]
# Strategies


# In[48]:


# Stats
from datetime import date
Strategy_Years = (Strategies.index[Strategies.shape[0]-1] - Strategies.index[0]).days/365
Strategy_Returns = Strategies.apply(lambda x: (1+x).prod()/Strategy_Years,axis=0)
Strategy_StDev = Strategies.apply(lambda x: np.std(x)*(12**0.5),axis=0)
Strategy_Sharpe = (Strategy_Returns-rf)/Strategy_StDev


# In[49]:


# def SortinoRatio(df, T):
#     temp = np.minimum(0, df - T)**2
#     temp_expectation = np.mean(temp)
#     downside_dev = np.sqrt(temp_expectation)
#     sortino_ratio = np.mean(df - T) / downside_dev
#     return(sortino_ratio)
# Strategy_Sortino = Strategies.apply(lambda x: SortinoRatio(x,np.mean(Strategies['Benchmark'])),axis=0)


# In[55]:


def MaxDrawdown(a):
    a = (1+a).cumprod()*100
    max_months = 0
    high_to_low = 0
    low_index = 0
    for i in range(len(a)-1):
        months = 0
        low = a[i]
        for j in range(i+1,len(a)):
            if a[j]<a[i]:
                months= months+1
                if low>a[j]:
                    low = a[j]
                    low_index = j
            else:
                break
        if months>max_months:
            max_months = months
        if (low-a[j])/a[j]<high_to_low:
            high_to_low = (low-a[j])/a[j]
    return max_months,high_to_low,low_index
Strategy_MaxDrawDown = Strategies.apply(lambda x: MaxDrawdown(x)[1],axis=0)
Strategy_MaxDrawDownMonths = Strategies.apply(lambda x: MaxDrawdown(x)[0],axis=0)
low_index = Strategies.apply(lambda x: MaxDrawdown(x)[0],axis=0)


# In[89]:


Strategy_Beta = Strategies.apply(lambda x: np.cov(x,Strategies['Benchmark'])[1,0]/np.var(Strategies['Benchmark']),axis = 0)


# In[90]:


VaR_90 = Strategies.apply(lambda x: np.quantile(x,0.1),axis = 0)


# In[91]:


import tabulate as tb


# In[94]:


Strategy_Stats = pd.DataFrame(['Benchmark','Long Only Equal Weight','Long Only Risk Parity','Market Neutral Equal Weight','Risk Budgeting',
        'Long Only Risk Budgeting','Long Only Equal Weight or Cash','Long Only Risk Parity or Cash',
        'Market Neutral Equal Weight or Cash','Risk Budgeting or Cash','Long Only Risk Budgeting or Cash'],columns = ['Strategy'])
Strategy_Stats['Returns'] = np.round(Strategy_Returns.reset_index(drop=True),2)
Strategy_Stats['St Dev'] = np.round(Strategy_StDev.reset_index(drop=True),2)
Strategy_Stats['Sharpe'] = np.round(Strategy_Sharpe.reset_index(drop=True),2)
Strategy_Stats['Max Drawdown'] = np.round(Strategy_MaxDrawDown.reset_index(drop=True),2)
Strategy_Stats['Max Drawdown Months'] = np.round(Strategy_MaxDrawDownMonths.reset_index(drop=True),2)
Strategy_Stats['Beta'] = np.round(Strategy_Beta.reset_index(drop=True),2)
Strategy_Stats['Var 90%'] = np.round(VaR_90.reset_index(drop=True),2)
Strategy_Stats


# In[107]:


with pd.ExcelWriter('C:\\Users\\dipan\\OneDrive\\Desktop\\Momentum Strategies.xlsx') as writer:
    Strategies.to_excel(writer, sheet_name='Strategies')
    Strategy_Stats.to_excel(writer, sheet_name='Strategy Stats')


# In[ ]:





# In[ ]:





# In[ ]:




