import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

accuracies = [91.8, 90.7, 89.5, 94.2, 95.7, 96.6, 92.2]
epochs = [1, 2, 3, 4, 5, 6, 7]

ynew = np.linspace(min(accuracies), max(accuracies), 300) #300 represents number of points to make between T.min and T.max

power_smooth = spline(epochs, accuracies, ynew)

plt.plot(power_smooth, ynew)
plt.show()



