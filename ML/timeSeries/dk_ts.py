import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data1 = pd.read_csv('input/filtered_sample_output.csv')
data = pd.read_csv('input/filtered_train_data.csv')
data_proc = pd.DataFrame()
data_proc['nuser_id'] = data['nuser_id']
data_proc['ncategory'] = data['ncategory']
data_proc['quantity'] = data['quantity']
#data_proc.index = data['date'] 
data_proc['date'] = data['date']

dfg = data_proc.groupby(['nuser_id','date', 'ncategory'])['quantity'].agg('sum').reset_index()
dfg = pd.DataFrame(dfg)

print (dfg['quantity'].std(), dfg['quantity'].mean(), dfg['quantity'].min(), dfg['quantity'].max())

catgCluster = pd.DataFrame()
catgCluster['quantity'] =dfg['quantity']
catgCluster['ncategory'] = dfg['ncategory'].astype(str)
#plt.rcParams.update({'figure.figsize': (10,10)})
plt.plot(catgCluster['quantity'])
plt.show()
#catgCluster.set_index('ncategory', inplace=True)



#print (catgCluster)
kmeans = KMeans(n_clusters=4).fit(catgCluster)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(catgCluster['ncategory'],catgCluster['quantity'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(catgCluster['quantity'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

sec_cat_indx = np.where(kmeans.labels_==2)[0] 
cat = list()
user = list()
date = list()
cat_df = pd.DataFrame()

for i in sec_cat_indx :
    cat.append(dfg['ncategory'].values[i])
    user.append(dfg['nuser_id'].values[i])
    date.append(dfg['date'].values[i])

cat_df['ncategory'] = cat
cat_df['nuser_id'] = user
cat_df['date'] = date
sec_cat_df = cat_df

frst_cat_indx = np.where(kmeans.labels_==3)[0]
cat = list()
user = list()
date = list()
cat_df = pd.DataFrame()
for i in frst_cat_indx :
    cat.append(dfg['ncategory'].values[i])
    user.append(dfg['nuser_id'].values[i])
    date.append(dfg['date'].values[i])

cat_df['ncategory'] = cat
cat_df['nuser_id'] = user
cat_df['date'] = date
frst_cat_df = cat_df


dfg2 = pd.DataFrame(frst_cat_df.groupby(['nuser_id'])['date'].agg('count').reset_index())

'''
df_usr = pd.DataFrame()
df_usr['date_num'] =  dfg2['date']
df_usr['nuser_id'] = dfg2['nuser_id'].astype(str)
df_usr.set_index('nuser_id', inplace=True)
#plt.rcParams.update({'figure.figsize': (10,10)})
plt.plot(df_usr['date_num'])
plt.show()


# Suggest: 4 categoris for data:
# 1: 0-10
# 2: 10-20
# 3: 20-30
# 4: >30

#print (catgCluster)
kmeans2 = KMeans(n_clusters=4).fit(df_usr)
centroids2 = kmeans2.cluster_centers_
print(centroids2)

plt.scatter(df_usr['nuser_id'],df_usr['date_num'], c= kmeans2.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(catgCluster['quantity'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)

plt.scatter(centroids2[:, 0], centroids2[:, 1], c='red', s=50)
plt.show()
'''

loyal_users = dfg2[dfg2['date'] > 15]
frst_cat_df 

first_user_cluster =  data_proc[data_proc['nuser_id'].isin(loyal_users['nuser_id'].values)] 
first_user_cluster =  first_user_cluster[first_user_cluster['ncategory'].isin(frst_cat_df['ncategory'].values)] 
first_user_cluster
