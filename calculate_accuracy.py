from sklearn.metrics import accuracy_score
from batch_perceptron import batch_perceptron
from online_training import online_training


def calculate_accuracy(x_without_bias, y_without_bias, algorithm):
    """
    Function to calculate the accuracy of generated model.
    
    It generates model using training data and calculate the 
    accuracy over testing data.
    """

    weights = []

    # Get only 75% of the random data to be used as a training data to generate the model
    training_data_x = x_without_bias[0: int(len(x_without_bias) *0.75), :]
    training_data_y = y_without_bias[0: int(len(y_without_bias) *0.75)]

    # Remaining data (25%) will be used to test our model
    testing_data_x = x_without_bias[int(len(x_without_bias) *0.75): len(x_without_bias), :]
    testing_data_y = y_without_bias[int(len(y_without_bias) *0.75): len(y_without_bias)]

    if algorithm == "Online Training" or algorithm == "OT":

        W_OT_random, updates_OT_random, epochs_OT_random, deltas_OT \
            = online_training(training_data_x, training_data_y)

        # Calculate the predicted output of testing data after generating weights
        y_model = testing_data_x[:, 0] * W_OT_random[0] + testing_data_x[:, 1] * W_OT_random[1] 

        weights =  W_OT_random 

    elif algorithm == "Batch Perceptron" or algorithm == "BP":

        W_BP_random, updates_BP_random, epochs_BP_random, deltas_BP \
            = batch_perceptron(training_data_x, training_data_y)

        # Calculate the predicted output of testing data after generating weights 
        y_model = testing_data_x[:, 0] * W_BP_random[0] + testing_data_x[:, 1] * W_BP_random[1] 

        weights =  W_BP_random


    else: 

        print("Unknown Algorithm")
        return
    
    # Convert positive values in y_model to 1s
    mask_for_y_positive = y_model >= 0
    y_model[mask_for_y_positive] = 1

    # Convert negative values in y_model to -1s
    mask_for_y_negative = y_model < 0
    y_model[mask_for_y_negative] = -1

    # Calculate the accuracy between actual and predicted data (y_model)
    accuracy = accuracy_score(y_model, testing_data_y)

    return accuracy, weights
