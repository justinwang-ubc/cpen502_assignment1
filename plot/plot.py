
import matplotlib.pyplot as plt
import numpy as np

y = np.loadtxt("E:\java\Assignment1\Data\ErrorVsEpochs-Binary-NumEpoch-2360.txt")

# plotting the points 
plt.plot(y)

# naming the x axis 
plt.xlabel('epoch')
# naming the y axis 
plt.ylabel('total error')

# giving a title to my graph
plt.title(' Binary Representation ')

# function to show the plot 
plt.show() 