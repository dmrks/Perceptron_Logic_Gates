import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


data = [[0, 0],[0, 1],[1, 0],[1, 1]]
labels = [0,1,1,1]
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))

#3

plt.scatter([point[0] for point in data],labels,c = labels)
classifier = Perceptron(max_iter = 40)
classifier.fit(data,labels)
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(i) for i in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
print(classifier.score(data,labels))
heatmap =plt.pcolormesh(x_values,y_values,distances_matrix)
plt.colorbar(heatmap)
plt.show()