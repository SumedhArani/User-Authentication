import numpy as np
from nnet import create_profile

training_data = list(zip(np.array([[12,21,3], [11, 20, 3]]), np.array([[0,1], [0,1]])))
net = create_profile('S')
test_data = list(zip(np.array([[13,22,3]]), np.array([1])))
net.SGD(training_data, 10, 3.0) #train the neural network
print(net.evaluate(test_data))
