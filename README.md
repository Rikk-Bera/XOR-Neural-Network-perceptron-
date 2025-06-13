# XOR-Neural-Network-perceptron-
This project implements a simple neural network using NumPy to learn the XOR logic function. It includes forward propagation, backpropagation, and gradient descent to train the model, with real-time cost visualization using Matplotlib.

Problem Statement
The XOR function is not linearly separable, making it a classic problem to demonstrate the capabilities of a multi-layer neural network. The network is trained to map the following inputs to their respective outputs:

Input:      Output:

[0, 0]  →    1  
[0, 1]  →    0  
[1, 0]  →    0  
[1, 1]  →    1

**Model Architecture**

Input Layer: 2 neurons (for each input feature)

Hidden Layer: 5 neurons with sigmoid activation

Output Layer: 1 neuron with sigmoid activation

**Dependencies**

Python 3.x
NumPy
Matplotlib

**Install using:**

pip install numpy matplotlib
**How It Works**

Weights are initialized randomly.
A forward pass computes the activations through the network.
A backward pass computes gradients via backpropagation.
Parameters are updated using gradient descent.

Training Visualization
The cost (binary cross-entropy loss) over 20,000 epochs is plotted to visualize convergence.
