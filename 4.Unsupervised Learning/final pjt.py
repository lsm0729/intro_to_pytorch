import pickle
import matplotlib.pyplot as plt
with open("kmean_score.pickle", "rb") as fr:
    scores = pickle.load(fr)

import numpy as np
#print(len(scores))



x= np.arange(1,21,1)





plt.plot(x,scores,marker='*')
plt.xticks(x)

plt.show()