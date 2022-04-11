import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
x_size = 10
y_size = 10
z_size = 2
M = np.zeros(shape=(x_size, y_size, z_size))
'''
#8
M[2][1][1] = 1
M[3][1][1] = 1
M[3][1][0] = 1
M[1][2][1] = 1
M[3][2][1] = 1
M[4][2][0] = 1
M[4][2][1] = 1
M[1][3][0] = 1
M[1][3][1] = 1
M[2][3][1] = 1
M[4][3][1] = 1
M[2][4][0] = 1
M[2][4][1] = 1
M[3][4][1] = 1
#----------------
#3
M[2][1][1] = 1
M[1][2][1] = 1
'''
#----------------
#6
M[2][1][1] = 1
M[3][1][1] = 1
M[1][2][1] = 1
M[3][2][0] = 1
M[1][3][1] = 1
M[2][3][0] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
counter_x = range(x_size)
counter_y = range(y_size)
counter_z = range(z_size)
x,y,z = np.meshgrid(counter_x, counter_y, counter_z)
ax.scatter(x,y,z, c=M.flat)


plt.show()