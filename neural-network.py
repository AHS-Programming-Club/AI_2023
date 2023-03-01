# This NN (neural network) model has 5 nodes (or node-like objs)
# 3 inputs, 1 node in a hidden layer, 1 output
'''
A - +
    ↓
B - D -> F
    ↑
C - +

A, B, C are inputs
D is the hidden node
F is the output node
'''


import random # used for random edge weights

# constant
e = 2.71828

# normalization/restriction function
# looks like a logarithmic growth function
def sigmoid(x):
    return 1/(1+e**(-x))

# "confidence level" of the output it provides
# this function looks like a bell curve
def sigmoid_deriv(x):
    return x * (1 - x)

# dot product, used to calculate the convolution/activation thing for each node
def dot(arr, arrt):
    s = 0
    n = len(arr)

    # assume equal lengths
    for i in range(n):
        s += (arr[i]*arrt[i])

    return s



# random seed yay
# the random.random() with the extra operations 
# assigns 3 edges (from a, b, c => d) randomized weights between -1 and 1
synaptic_weights = [2 * random.random() - 1 for i in range(3)]
print('starting weights', synaptic_weights) # for reference

# training data used
# the "answer" is always the middle value
training_data = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
training_output = [[0], [0], [1], [0], [1], [1]]

# data used for testing
'''
test_data = [[1, 0, 0], [0, 1, 0]]
test_out = [[0], [1]]
'''


# 10k training times!
# gotta get an accurate NN :/
for iter in range(10000):
    # train it to make it better
    
    for idx in range(len(training_data)):
        # Pass the training set through our neural network (a single neuron).

        # this is our result (that the neuron tells us)
        # aka forward propogation
        output = sigmoid(dot(training_data[idx], synaptic_weights))

        # Calculate the error (The difference between the desired output and the predicted output).
        error = training_output[idx][0] - output
    
        # back propogation
        # change the confidence weights
        # think about this like "learning from your mistakes" but for the NN
        # changing the weights can make it value one input over the other
        synaptic_weights[0] += training_data[idx][0]*error*sigmoid_deriv(output)
        synaptic_weights[1] += training_data[idx][1]*error*sigmoid_deriv(output)
        synaptic_weights[2] += training_data[idx][2]*error*sigmoid_deriv(output)
        pass
    
    continue


print("\nend weights:", synaptic_weights)
# test cases
print("Considering new situation [1, 0, 0] -> ?: ")
print(sigmoid(dot([1, 0, 0], synaptic_weights)))

print("\n")
print("Considering new situation [0, 1, 0] -> ?: ")
print(sigmoid(dot([0, 1, 0], synaptic_weights)))

