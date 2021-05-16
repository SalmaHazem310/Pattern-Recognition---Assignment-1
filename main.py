import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from batch_perceptron import batch_perceptron
from online_training import online_training
from calculate_accuracy import calculate_accuracy
from model_visualization import model_visualization


# ----------------------------------------------------------------------------------------------
# Problem 1 with bias

x_problem1 = np.array([[50, 55, 70, 80, 130, 150, 155, 160], [1, 1, 1, 1, 1, 1, 1, 1]]).T
y_problem1 = np.array([1, 1, 1, 1, -1, -1, -1, -1])


# ----------------------------------------------------------------------------------------------
# Problem 4 with bias

x_problem4 = np.array([[0, 255, 0, 0, 255, 0, 255, 255], [0, 0, 255, 0, 255, 255, 0, 255],
                [0, 0, 0, 255, 0, 255, 255, 255], [1, 1, 1, 1, 1, 1, 1, 1]]).T
y_problem4 = np.array([1, 1, 1, -1, 1, -1, -1, 1])


# ----------------------------------------------------------------------------------------------
def random_x_and_y():
    """
    Generate random data.
    """

    x, y = make_classification(25, n_features= 2, n_redundant= 0, n_informative= 1, n_clusters_per_class= 1)

    mask_for_y = y == 0
    y[mask_for_y] = -1

    ax = fig.add_subplot(311)
    ax.title.set_text('Random Data Without Bias')

    plt.scatter(x[:, 0], x[:, 1], marker='o', c= y, s= 25, edgecolor='k')

    return x, y


# ----------------------------------------------------------------------------------------------
def subplots(number, data, title):
    """
    Function to plot data in one figure.
    """
    plt.subplot(2, 2, number).title.set_text(title)
    plt.plot(data)
    

# ----------------------------------------------------------------------------------------------
# Apply Algorthims and Plot Deltas


# Apply online training algorithm on problems 1 and 4
W_online_training_p1, updates_online_training_p1, epochs_online_training_p1, deltas_OT_p1 \
    = online_training(x_problem1, y_problem1)
W_online_training_p4, updates_online_training_p4, epochs_online_training_p4, deltas_OT_p4 \
    = online_training(x_problem4, y_problem4)

# Apply batch perceptron algorithm on problems 1 and 4
W_batch_preceptron_p1, updates_batch_preceptron_p1, epochs_batch_preceptron_p1, deltas_BP_p1 \
    = batch_perceptron(x_problem1, y_problem1)
W_batch_preceptron_p4, updates_batch_preceptron_p4, epochs_batch_preceptron_p4, deltas_BP_p4 \
    = batch_perceptron(x_problem4, y_problem4)


print('\n')
print("{:<22} {:<18} {:<8} {:<8}".format('Problem / Algorithm', 'Number of Updates', 'Epochs', 'Weights'))

print("{:<22} {:<18} {:<8} {:<8}".format('1 - Online Training', str(updates_online_training_p1), \
    str(epochs_online_training_p1), str(W_online_training_p1)), '\n')

print("{:<22} {:<18} {:<8} {:<8}".format('4 - Online Training', str(updates_online_training_p4), \
    str(epochs_online_training_p4), str(W_online_training_p4)), '\n')

print("{:<22} {:<18} {:<8} {:<8}".format('1 - Batch Perceptron', str(updates_batch_preceptron_p1), \
    str(epochs_batch_preceptron_p1), str(W_batch_preceptron_p1)), '\n')

print("{:<22} {:<18} {:<8} {:<8}".format('4 - Batch Perceptron', str(updates_batch_preceptron_p4), \
    str(epochs_batch_preceptron_p4), str(W_batch_preceptron_p4)), '\n')


# Plot deltats of each
subplots(1, deltas_OT_p1, 'Deltas OT P1')
subplots(2, deltas_OT_p4, 'Deltas OT P4')
subplots(3, deltas_BP_p1, 'Deltas BP P1')
subplots(4, deltas_BP_p4, 'Deltas BP P4')


# ----------------------------------------------------------------------------------------------
# Accuracy and Model Visualization


fig = plt.figure()

# Generate random data and assign it to x_without_bias and y_without_bias
x_without_bias, y_without_bias = random_x_and_y()


# Calculate accuracy of each algorithm 
accuracy_OT, W_OT = calculate_accuracy(x_without_bias, y_without_bias, 'OT')
print("Online Training Accuracy: ", accuracy_OT)

accuracy_BP, W_BP = calculate_accuracy(x_without_bias, y_without_bias, 'BP')
print("Batch Perceptron Accuracy: ", accuracy_BP)


# Visualize model of each algorithm
model_visualization(x_without_bias, y_without_bias, W_OT, 'OT', fig)
model_visualization(x_without_bias, y_without_bias, W_BP, 'BP', fig)

plt.show()

