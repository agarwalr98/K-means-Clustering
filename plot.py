import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import random

print("========== K MEANS CLUSTERING ===========\n")
colors=['red', 'blue', 'green']

TotalData = 0
with open("Data/NonLinear/group03.txt") as myfile:
    next(myfile)
    for i in myfile:
        TotalData = TotalData + 1

index = 0
Data = np.zeros(shape = (int(TotalData), int(2)), dtype = float)

with open("Data/NonLinear/group03.txt") as myfile:
    next(myfile)
    for i in myfile:
        i = i.split()
        Data[index][0] = float(i[0])
        Data[index][1] = float(i[1])
        index = index + 1

# print(Data)
index = 0
for i in Data:
    if index<300:
        plt.plot(float(i[0]), float(i[1]), "o", color=colors[0])
    elif index >=300 and index <800:
        plt.plot(float(i[0]), float(i[1]),  "o", color=colors[1])
    else:
        plt.plot(float(i[0]), float(i[1]), "o",color =colors[2])
    index = index + 1

points = [Line2D([0], [0], color=c) for c in colors]
labels = ['Class 1', 'Class 2', 'Class 3']
plt.legend(points, labels)
plt.show()
