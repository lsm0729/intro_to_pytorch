import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import show_images, show_images_by_digit, fit_random_forest_classifier2
from helper_functions import fit_random_forest_classifier, do_pca, plot_components
import test_code as t

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


"""
1. Use pandas to read in the dataset, which can be found at the following address './data/train.csv'. 
Take a look at info about the data using head, tail, describe, info, etc. 
You can learn more about the data values from the article here: https://homepages.inf.ed.ac.uk/rbf/HIPR2/value.htm.
"""

train = pd.read_csv('train.csv')
train.fillna(0, inplace=True)

"""
2. Create a vector called y** that holds the **label column of the dataset. 
Store all other columns holding the pixel data of your images in X.
"""

# save the labels to a Pandas series target
y = train['label']

# Drop the label feature
X = train.drop("label",axis=1)

"""
3. Now use the show_images_by_digit function from the helper_functions module to take a look some of the 1's, 2's, 3's, 
or any other value you are interested in looking at. Do they all look like what you would expect?
"""

#Check Your Solution
t.question_two_check(y, X)


#show_images_by_digit(2)


fit_random_forest_classifier(X, y)