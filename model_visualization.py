import numpy as np
import matplotlib.pyplot as plt


def model_visualization(data_x, data_y, W, Algorithm, fig):

    # Get classification boundary (W1x + W0 = 0)
    m = -W[0] / W[1]
    x = np.linspace(-5, 5, num= 100)
    y = m * x

    if Algorithm == "OT":

        ax1 = fig.add_subplot(312)
        ax1.title.set_text('Online Training')

    elif Algorithm == "BP":
        
        ax2 = fig.add_subplot(313)
        ax2.title.set_text('Batch Perceptron')

    # Plot the model over data
    plt.scatter(data_x[:, 0], data_x[:, 1], marker='o', c= data_y, s= 25, edgecolor='k')
    plt.plot(x, y, 'k-')
    
