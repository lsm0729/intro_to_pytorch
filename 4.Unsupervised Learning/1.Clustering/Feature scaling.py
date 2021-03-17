import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing as p
import helpers2 as h
import tests as t

plt.rcParams['figure.figsize'] = (16, 9)


# Create the dataset for the notebook
data = h.simulate_data(200, 2, 4)
df = pd.DataFrame(data)
df.columns = ['height', 'weight']
df['height'] = np.abs(df['height']*100)
df['weight'] = df['weight'] + np.random.normal(50, 10, 200)




#plt.scatter(df['height'], df['weight'])
#plt.show()



##TODO: apply scaling.. minmaxscaler and StandardScaler

df_ss = p.StandardScaler().fit_transform(df) # Fit and transform the data

df_ss = pd.DataFrame(df_ss) #create a dataframe

df_ss.columns = ['height', 'weight'] #add column names again


#plt.scatter(df_ss['height'], df_ss['weight']); # create a plot
#plt.show()



df_mm = p.MinMaxScaler().fit_transform(df) # fit and transform
df_mm = pd.DataFrame(df_mm) #create a dataframe
df_mm.columns = ['height', 'weight'] #change the column names

#plt.scatter(df_mm['height'], df_mm['weight']); #plot the data
#plt.show()


def fit_kmeans(data, centers):
    '''
    INPUT:
        data = the dataset you would like to fit kmeans to (dataframe)
        centers = the number of centroids (int)
    OUTPUT:
        labels - the labels for each datapoint to which group it belongs (nparray)

    '''
    kmeans = KMeans(centers)
    labels = kmeans.fit_predict(data)
    return labels


labels = fit_kmeans(df, 10)  # fit kmeans to get the labels

# Plot the original data with clusters
#plt.scatter(df['height'], df['weight'], c=labels, cmap='Set1')
#plt.show()


labels = fit_kmeans(df_mm, 10)  # fit kmeans to get the labels

# plot each of the scaled datasets
#plt.scatter(df_mm['height'], df_mm['weight'], c=labels, cmap='Set1')
#plt.show()




labels = fit_kmeans(df_ss, 10)

plt.scatter(df_ss['height'], df_ss['weight'], c=labels, cmap='Set1');
plt.show()


