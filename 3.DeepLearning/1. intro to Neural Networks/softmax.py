
import cupy as np

def softmax(L):

    expL = np.exp(L)

    return np.divide(expL,expL.sum())


sample_list = [2,1,0]

#print(softmax(sample_list))

a= np.array([1,2,3,4,5])

print(np.log(a))

