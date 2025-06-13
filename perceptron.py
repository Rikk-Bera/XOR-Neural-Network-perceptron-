import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x)/((np.exp(-x)+1)**2)
def cost_computation(y, yhat):
    logprobs = -(1/m)*np.sum(np.multiply(y, np.log(yhat)) + np.multiply((1-y), np.log(1-yhat)))
    return logprobs
lr = 0.1
xor_input = np.array([[0,0],[0,1],[1,0],[1,1]])
xor_output = np.array([[1,0,0,1]])

X = xor_input.T
Y = xor_output
print(X.shape)
output_dim = len(Y.T)
print(output_dim)
n0, m = X.shape
n1 = 5
W1 = np.random.random((n1,n0))
b1 = np.zeros((n1,1))

n2 = 1
W2 = np.random.random((n2,n1))
b2 = np.zeros((n2,1))
NumOfEpochs = 20000
Cost = []

for epoch in range(NumOfEpochs):
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    Cost.append(np.squeeze(cost_computation(Y,A2)))
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2), (A1*(1-A1)))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    W1 = W1 - lr*dW1
    W2 = W2 - lr*dW2
    b1 = b1 - lr*db1
    b2 = b2 - lr*db2
print(A2)
def predict(W1,W2,b1,b2,X):
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    predictions = np.round(A2)
    return predictions
import matplotlib.pyplot as plt
Cost = np.array(Cost)
print(Cost.shape)
plt.plot(Cost)
plt.show()