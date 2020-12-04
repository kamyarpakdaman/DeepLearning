# In this program, we use perceptrons to model the logic gates of computers.

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# This function trains a perceptron with AND logical training dataset and returns the training data along with decision function results.

def and_trainer():

    # AND gate possible 2D inputs and outputs.

    data_and = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels_and = [0, 0, 0, 1]

    # Now we build a perceptron to learn AND.

    classifier_and = Perceptron(max_iter = 40)
    classifier_and.fit(data_and, labels_and)
    # print(classifier_and.score(data_and, labels_and))

    # Even though an input like [0.5, 0.5] isn’t a real input to an AND logic gate, to see the distances of the input points to the separator 
    # line we can use the decision function.

    x_and = np.linspace(0, 1, 100)
    y_and = np.linspace(0, 1, 100)

    point_grid_and = list(product(x_and, y_and))

    distances_and = classifier_and.decision_function(point_grid_and)

    # distances contains positive and negative values, yet we want just the absolute distances.

    abs_distances_and = [abs(i) for i in distances_and]
    distances_matrix_and = np.reshape(abs_distances_and, (100, 100))

    return x_and, y_and, distances_matrix_and

# This function trains a perceptron with OR logical training dataset and returns the training data along with decision function results.

def or_trainer():

    # OR gate possible 2D inputs and outputs.

    data_or = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels_or = [0, 1, 1, 1]

    # Now we build a perceptron to learn OR.

    classifier_or = Perceptron(max_iter = 40)
    classifier_or.fit(data_or, labels_or)
    # print(classifier_or.score(data_or, labels_or))

    # Even though an input like [0.5, 0.5] isn’t a real input to an OR logic gate, to see the distances of the input points to the separator 
    # line we can use the decision function.

    x_or = np.linspace(0, 1, 100)
    y_or = np.linspace(0, 1, 100)

    point_grid_or = list(product(x_or, y_or))

    distances_or = classifier_or.decision_function(point_grid_or)

    # distances contains positive and negative values, yet we want just the absolute distances.

    abs_distances_or = [abs(i) for i in distances_or]
    distances_matrix_or = np.reshape(abs_distances_or, (100, 100))

    return x_or, y_or, distances_matrix_or

# This function trains a perceptron with XOR logical training dataset and returns the training data along with decision function results.

def xor_trainer():
    
    # XOR gate possible 2D inputs and outputs.

    data_xor = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels_xor = [0, 1, 1, 0]

    # Now we build a perceptron to learn XOR.

    classifier_xor = Perceptron(max_iter = 40)
    classifier_xor.fit(data_xor, labels_xor)
    # print(classifier_xor.score(data_xor, labels_xor))

    # Even though an input like [0.5, 0.5] isn’t a real input to an OR logic gate, to see the distances of the input points to the separator 
    # line we can use the decision function.

    x_xor = np.linspace(0, 1, 100)
    y_xor = np.linspace(0, 1, 100)

    point_grid_xor = list(product(x_xor, y_xor))

    distances_xor = classifier_xor.decision_function(point_grid_xor)

    # distances contains positive and negative values, yet we want just the absolute distances.

    abs_distances_xor = [abs(i) for i in distances_xor]
    distances_matrix_xor = np.reshape(abs_distances_xor, (100, 100))

    return x_xor, y_xor, distances_matrix_xor

x_and, y_and, distances_matrix_and = and_trainer()
x_or, y_or, distances_matrix_or = or_trainer()
x_xor, y_xor, distances_matrix_xor = xor_trainer()

# Now we draw three heat maps for AND, OR, and XOR decision functions using matplotlib pcolormesh().

plt.figure()

heatmap_and = plt.pcolormesh(x_and, y_and, distances_matrix_and)
ax = plt.gca()
cbar = plt.colorbar(heatmap_and)
cbar.ax.tick_params(size = 0)
plt.title('AND perceptron decision function')
plt.tick_params(left = False, right = False, bottom = False, top = False)
plt.savefig('result_1.png')

plt.show()

plt.figure()

heatmap_or = plt.pcolormesh(x_or, y_or, distances_matrix_or)
ax = plt.gca()
cbar = plt.colorbar(heatmap_and)
cbar.ax.tick_params(size = 0)
plt.title('OR perceptron decision function')
plt.tick_params(left = False, right = False, bottom = False, top = False)
plt.savefig('result_2.png')

plt.show()

plt.figure()

heatmap_xor = plt.pcolormesh(x_xor, y_xor, distances_matrix_xor)
ax = plt.gca()
cbar = plt.colorbar(heatmap_and)
cbar.ax.tick_params(size = 0)
plt.title('XOR perceptron decision function')
plt.tick_params(left = False, right = False, bottom = False, top = False)
plt.savefig('result_3.png')

plt.show()

# Note that while the separation lines are obvious for AND and OR perceptrons, the separation line isn't visible for XOR
# because there is no such line. In fact, XOR problem isn't linearly separable, and hence, a single 2D perceptron cannot
# solve the classification problem with one line.

print('\nThanks for reviewing')

# Thanks for reviewing
