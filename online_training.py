import random
import numpy as np
import sys


def online_training(x, y):

    # Size of given data
    size_data = len(x)

    # Number of features 
    size_x = x.shape[1]

    # Generate random weights of size equals to number of features (including bias) 
    W = np.random.uniform(-1, 1, size = size_x)

    # Idetify a very small value to compare it with norm of delta
    e = sys.float_info.epsilon

    # Identify delta to update weights
    # Initially, assigned with ones to get into While loop
    delta = np.ones(size_x)

    number_updates = 0
    epochs = 0
    delta_means = []

    # Loop over data until deltas becomes very small
    while np.linalg.norm(delta, 1) > e:

        delta = np.zeros(size_x)

        # Loop over each point
        for i in range(size_data):

            # Calculate the predicted output
            WX = W.dot(x[i])
     
            # Compare the predicted and actual output
            if(y[i] * WX <= 0):

                # Update weights at each observation
                delta = delta - (y[i]*x[i])
                delta = delta/size_data
                delta_means.append(np.mean(delta))
                W = W - delta

                number_updates += 1

        epochs += 1

    return W, number_updates, epochs, delta_means

