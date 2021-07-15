import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import itertools 
#import statsmodels.api as sm 
import warnings
import numpy as np
#import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
#import pandas as pd
#import matplotlib
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller


#data1 = pd.read_csv('input/filtered_sample_output.csv')
data = pd.read_csv('datasets/filtered_train_data.csv')

#Removing un-necessary items
data_proc = pd.DataFrame()
data_proc['nuser_id'] = data['nuser_id']
data_proc['ncategory'] = data['ncategory']
data_proc['quantity'] = data['quantity']
data_proc['date'] = data['date']

# STEP1: Finding most sold categories
# Grouping users, and n_cats. then get sum of quantities of products 
# in each category
dfg = data_proc.groupby(['nuser_id','date', 'ncategory'])['quantity'].agg('sum').reset_index()
dfg = pd.DataFrame(dfg)
print (dfg['quantity'].std(), dfg['quantity'].mean(), dfg['quantity'].min(), dfg['quantity'].max())


# Clustering Most Sold products using K-means
catgCluster = pd.DataFrame()
catgCluster['quantity'] =dfg['quantity']
catgCluster['ncategory'] = dfg['ncategory'].astype(str)
plt.plot(catgCluster['quantity'])
#plt.show()

kmeans = KMeans(n_clusters=4).fit(catgCluster)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(catgCluster['ncategory'],catgCluster['quantity'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.show()

# Extracting data related to First cluster (most sold categories)
frst_cat_indx = np.where(kmeans.labels_==3)[0]
cat = list(); user = list(); date = list()
cat_df = pd.DataFrame()
for i in frst_cat_indx :
    cat.append(dfg['ncategory'].values[i])
    user.append(dfg['nuser_id'].values[i])
    date.append(dfg['date'].values[i])

cat_df['ncategory'] = cat ;cat_df['nuser_id'] = user; cat_df['date'] = date
frst_cat_df = cat_df


# Extracting data related to second cluster (most sold categories)
sec_cat_indx = np.where(kmeans.labels_==2)[0] 
cat = list(); user = list();date = list()
cat_df = pd.DataFrame()

for i in sec_cat_indx :
    cat.append(dfg['ncategory'].values[i])
    user.append(dfg['nuser_id'].values[i])
    date.append(dfg['date'].values[i])

cat_df['ncategory'] = cat; cat_df['nuser_id'] = user ; cat_df['date'] = date
sec_cat_df = cat_df


# STEP2: Finding the most loyal users on the Most Sold products
#dfg2 = pd.DataFrame(frst_cat_df.groupby(['nuser_id'])['date'].agg('count').reset_index())

dfg2 =  pd.DataFrame(data_proc.groupby(['nuser_id'])['date'].agg('count').reset_index())

# Clustering Loyal users using K-means
df_usr = pd.DataFrame()
df_usr['date_num'] =  dfg2['date']
df_usr['nuser_id'] = dfg2['nuser_id'].astype(str)
#df_usr.set_index('nuser_id', inplace=True)
plt.plot(df_usr['date_num'])
#plt.show()

kmeans2 = KMeans(n_clusters=4).fit(df_usr)
centroids2 = kmeans2.cluster_centers_
print(centroids2)
#plt.scatter(df_usr['nuser_id'],df_usr['date_num'], c= kmeans2.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(catgCluster['quantity'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids2[:, 0], centroids2[:, 1], c='red', s=50)
#plt.show()


# Extracting data related to the First cluster (most loyal users)
frst_usr_indx = np.where(kmeans2.labels_==0)[0]
user = list()
usr_df = pd.DataFrame()
for i in frst_usr_indx :
    user.append(dfg['nuser_id'].values[i])
usr_df['nuser_id'] = user
#loyal_users = dfg2[dfg2['date'] > 15]
#frst_cat_df 

# Step3: Making Loyal users list (with the most sold categories)
loyal_users =  data_proc[data_proc['nuser_id'].isin(usr_df['nuser_id'].values)] 
loyal_users =  loyal_users[loyal_users['ncategory'].isin(frst_cat_df['ncategory'].values)] 
print (len(loyal_users))


# Step3 Finding First Cluster of Loyal Users, on Most sold categories
'''dfg = data_proc.groupby(['nuser_id','date', 'ncategory'])['quantity'].agg('sum').reset_index()
dfg = pd.DataFrame(dfg)
first_user_cluster =  dfg[dfg['nuser_id'].isin(loyal_users['nuser_id'].values)] 
first_user_cluster =  first_user_cluster[first_user_cluster['ncategory'].isin(frst_cat_df['ncategory'].values)] 
first_user_cluster
'''

cat_unq = loyal_users['ncategory'].unique()
user_unq  = loyal_users['nuser_id'].unique()
print(len(loyal_users), len(cat_unq), len(user_unq))
print( len(usr_df))
print( len(frst_cat_df))

loyal_users = loyal_users.head(10)

# Step4: Call autoarima per user/cat 



def check_stationary(data, epsilon=0.05):      
    '''
    rolmean = pd.Series(data).rolling(window=12).mean()
    rolstd = pd.Series(data).rolling(window=12).std()
    Plot rolling statistics:
    
    plt.figure(figsize=(12, 8))
    orig = plt.plot(data, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    '''
    
    result = adfuller(data)
    '''
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))   
    '''
    if result[1] < epsilon :
        print('data is stationary')
        return True
    else:
        print('data is NOT stationary')
        return False
    
def make_stationary_method(data, intervals, method="diff"):
    if method=='MA': 
        moving_avg = pd.Series(data).rolling(interval).mean()
        stat_data = data - moving_avg
        stat_data.dropna(inplace=True) # first 6 is nan value due to window size            
    else:
        stat_data = data - data.shift()

    if not check_stationary(stat_data):
        print('It is still Not stationary!')
    '''
    plot_correlation(stat_data )
    plt.xlabel('Time')
    plt.ylabel('sale')
    plt.title('The made stationary data')
    stat_data.plot()
    plt.show()
    '''
    return stat_data 


arima_model = list()
for i in user_unq:
    temp_u = loyal_users[loyal_users['nuser_id']==i]
    if temp_u.empty :
        continue
    for j in cat_unq:
        temp_uc = temp_u[temp_u ['ncategory'] == j]
        if temp_uc.empty or  len(temp_uc.index)<10:
            continue
        temp_data = pd.DataFrame()
        temp_data['date'] = temp_uc['date']
        temp_data['quantity'] =  temp_uc['quantity']
        temp_data.set_index('date', inplace=True)
        
        if not check_stationary(temp_data):
            temp_data = make_stationary_method(temp_data,1) 
        try:
            model = pm.auto_arima(temp_data.values, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=32, max_q=2, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
            fc = model.predict(n_periods=60)
            if sum(fc) > 0:
                fc=1
            else:
                fc=0
            arima_model.append({'nuser_id':i,'ncategory':j, 'model': model, 'forecast': fc})
        except:
            continue

#print(arima_model['forcast'] )
