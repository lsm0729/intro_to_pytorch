import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from IPython.display import Image
from sklearn.datasets.samples_generator import make_blobs

# DSND colors: UBlue, Salmon, Gold, Slate
plot_colors = ['#02b3e4', '#ee2e76', '#ffb613', '#2e3d49']

# Light colors: Blue light, Salmon light
plot_lcolors = ['#88d0f3', '#ed8ca1', '#fdd270']

# Gray/bg colors: Slate Dark, Gray, Silver
plot_grays = ['#1c262f', '#aebfd1', '#fafbfc']


def create_data():
    n_points = 120
    X = np.random.RandomState(3200000).uniform(-3, 3, [n_points, 2])
    X_abs = np.absolute(X)

    inner_ring_flag = np.logical_and(X_abs[:, 0] < 1.2, X_abs[:, 1] < 1.2)
    outer_ring_flag = X_abs.sum(axis=1) > 5.3
    keep = np.logical_not(np.logical_or(inner_ring_flag, outer_ring_flag))

    X = X[keep]
    X = X[:60]  # only keep first 100
    X1 = np.matmul(X, np.array([[2.5, 0], [0, 100]])) + np.array([22.5, 500])

    plt.figure(figsize=[15, 15])

    plt.scatter(X1[:, 0], X1[:, 1], s=64, c=plot_colors[-1])

    plt.xlabel('5k Completion Time (min)', size=30)
    plt.xticks(np.arange(15, 30 + 5, 5), fontsize=30)
    plt.ylabel('Test Score (raw)', size=30)
    plt.yticks(np.arange(200, 800 + 200, 200), fontsize=30)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width=2)
    plt.savefig('C18_FeatScalingEx_01.png', transparent=True)

    data = pd.DataFrame(X1)
    data.columns = ['5k_Time', 'Raw_Test_Score']
    #plt.show()

    return data

data = create_data()


n_clusters = 2
model = KMeans(n_clusters=n_clusters)
preds = model.fit_predict(data)


def plot_clusters(data, preds, n_clusters):
    plt.figure(figsize=[15, 15])

    for k, col in zip(range(n_clusters), plot_colors[:n_clusters]):
        my_members = (preds == k)
        plt.scatter(data['5k_Time'][my_members], data['Raw_Test_Score'][my_members], s=64, c=col)

    plt.xlabel('5k Completion Time (min)', size=30)
    plt.xticks(np.arange(15, 30 + 5, 5), fontsize=30)
    plt.ylabel('Test Score (raw)', size=30)
    plt.yticks(np.arange(200, 800 + 200, 200), fontsize=30)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width=2)
    plt.show()

#plot_clusters(data, preds, 2)




##TODO:  Now create two new columns to add to your data dataframe.
##TODO: The first is test_scaled, which you should create by subtracting the mean test score and dividing by the standard deviation test score.

# your work here
data['test_scaled'] = (data['Raw_Test_Score'] - np.mean(data['Raw_Test_Score']))/np.std(data['Raw_Test_Score'])
data['5k_time_sec'] = data['5k_Time']*60


"""
Now, similar to what you did in question 2, instantiate a kmeans model with 2 cluster centers. 
Use your model to fit and predict the the group of each point in your dataset. 
Store the predictions in preds. If you correctly created the model and predictions, 
you should see a right (blue) cluster and left (pink) cluster when running the following cell.
"""
n_clusters = 2
model = KMeans(n_clusters = n_clusters)
preds = model.fit_predict(data)
plot_clusters(data, preds, n_clusters)
