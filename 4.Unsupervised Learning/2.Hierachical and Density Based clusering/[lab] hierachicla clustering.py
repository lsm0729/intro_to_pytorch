from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

iris = datasets.load_iris()

#print(iris.data[:10])
## iris.target contains the labels that indicate which type of Iris flower each sample is
#print(iris.target)



## ward method ( default of hierarchical clustering in sklearn
ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(iris.data)

##  case of complete and avg link


complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
complete_pred = complete.fit_predict(iris.data)

avg = AgglomerativeClustering(n_clusters=3, linkage='average')
avg_pred = avg.fit_predict(iris.data)


"""
To determine which clustering result better matches the original labels of the samples, 
we can use adjusted_rand_score which is an external cluster validation index which results in a score between -1 and 1, 
where 1 means two clusterings are identical of how they grouped the samples in a dataset 
(regardless of what label is assigned to each cluster).
Cluster validation indices are discussed later in the course.
"""

from sklearn.metrics import adjusted_rand_score

ward_ar_score = adjusted_rand_score(iris.target, ward_pred)

complete_ar_score = adjusted_rand_score(iris.target, complete_pred)

avg_ar_score = adjusted_rand_score(iris.target, avg_pred)


print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)





"""
3. The Effect of Normalization on Clustering

Can we improve on this clustering result?
Let's take another look at the dataset
"""





#print(iris.data[:15])

"""
Looking at this, we can see that the forth column has smaller values than the rest of the columns, 
and so its variance counts for less in the clustering process (since clustering is based on distance). 
Let us normalize the dataset so that each dimension lies between 0 and 1, so they have equal weight in the clustering process.

This is done by subtracting the minimum from each column then dividing the difference by the range.

sklearn provides us with a useful utility called preprocessing.normalize() that can do that for us
"""


from sklearn import preprocessing

normalized_X = preprocessing.normalize(iris.data)
#print(normalized_X[:10])

"""
Now all the columns are in the range between 0 and 1. 
Would clustering the dataset after this transformation lead to a better clustering? 
(one that better matches the original labels of the samples)
"""

ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(normalized_X)

complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
complete_pred = complete.fit_predict(normalized_X)

avg = AgglomerativeClustering(n_clusters=3, linkage="average")
avg_pred = avg.fit_predict(normalized_X)


ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)


"""
4. Dendrogram visualization with scipy
Let's visualize the highest scoring clustering result.

To do that, we'll need to use Scipy's linkage function to perform the clusteirng again 
so we can obtain the linkage matrix it will later use to visualize the hierarchy
"""

from scipy.cluster.hierarchy import linkage

linkage_type = 'ward'

linkage_matrix = linkage(normalized_X,method=linkage_type)

"""
Plot using scipy's dendrogram function
"""

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

#plt.figure(figsize=(18,50))

#dendrogram(linkage_matrix)
#plt.show()




"""
5. Visualization with Seaborn's clustermap

The seaborn plotting library for python can plot a clustermap, 
which is a detailed dendrogram which also visualizes the dataset in more detail. 
It conducts the clustering as well, so we only need to pass it the dataset and the linkage type we want, 
and it will use scipy internally to conduct the clustering
"""


import seaborn as sns

sns.clustermap(normalized_X, figsize=(30,50), method=linkage_type, cmap='viridis',annot=True)

# Expand figsize to a value like (18, 50) if you want the sample labels to be readable
# Draw back is that you'll need more scrolling to observe the dendrogram

plt.show()

"""
Looking at the colors of the dimensions can you observe how they differ between the three type of flowers? 
You should at least be able to notice how one is vastly different from the two others (in the top third of the image).
"""