# Report
* Doaa Sherif - 1170122
* Salma Hazem - 1170425

## Algorithms
It takes features and outputs data to generate its model after generating weights.   
We first assign weights random values between (-1, 1) and keep updates them until error becomes very small (negligible).   
Weights are updated after checking if both the actual outputs and the prdicted outputs (WX), are of the same sign, by multiplying them together, if the output is positive, they have the same signs (no update for weights). Otherwise, they have different signs (weights need update).

## Updates
### Online Training
Updates at once, as weights are updated at each error discovery (different signs).

### Batch Perceptron
Waits for a complete epoch (iterate over the whole dataset) to update the weights.

## Comparison
| Problem | Algorithm | Number of Updates | Epochs | Weights |
| ------- | ----------| ----------------- | ------ | ------- |
| 1 |Online Training| 203 | 67 | [-0.10530988 10.44270315] |
| 4 |Online Training| 14 | 8 | [ 56.20623563  51.52775353 -88.93819821   0.09300643] |
| 1 |Batch Perceptron| 2631 | 2631 | [ -6.08857691 525.79577087] |
| 4 |Batch Perceptron| 7 | 7 | [ 63.5786326   32.83898533 -63.84049943   0.24364096] |

* In Batch Perceptron Algorithm, number of updates is equal to the number of epochs
* Online training is more efficient than batch perceptron in large data set
    * In problem 1, online training is more efficient (reaches zero error first)
    * In problem 4, number of epoches using both algorithms is almost the same

## Accuracy
* make_classification function is used to generate random data 
* Training data is 75% of the generated data
* Testing data is the remaining 25% of the data
* Applying both algorithms on tarining data and generated the model
* Calculating the predicted output using the testing data
* Masking the predicted output
* Getting the accuracy between the outputs of the testing data (actual) and the predicted output

## Model Visualization
* Getting the classification boundary by solving this equation
    * W1x + W0 = 0
* Slope of the model
    * m = -W[0] / W[1]
* Multiplying the slope by the linespace (x) between -5 and 5 and getting y
* Plotting the output (x, y) over our generated data