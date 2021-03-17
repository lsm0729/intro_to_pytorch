import cupy as np


def sigmoid(x):

    return 1/(1+np.exp(-x))



n_input = 4
n_hidden = 3
n_output = 2


np.random.seed(4)


X=np.random.randn(4)

weights_input_to_hidden = np.random.normal(0,scale = 0.1, size=(n_input,n_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(n_hidden, n_output))


#TODO : 1 by 4 * 4 by 3
hidden_layer_in = np.dot(X,weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)


print('Hidden-layer Output:')
print(hidden_layer_out)

#TODO : 1 by 3 * 3 by 2
output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)