import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import csv

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 40.0, 0.01)

states = odeint(f, state0, t)

# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot(states[:, 0], states[:, 1], states[:, 2])
# plt.draw()
# plt.show()

autoCorr = signal.correlate(states[:, 0], states[:, 0], method='direct')
fig = plt.figure()
plt.plot(autoCorr)#[0:len(autoCorr)//10])
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()


minimums = signal.argrelextrema(states[:, 2], np.less)
print(minimums)
#
# writer = csv.writer(open("C:/Users/h_gho/SYSYTEM_DYNAMICS_WEB_APP/lorenz_data.csv", 'w',newline=''))
# for row in states:
#     # print(row)
#     writer.writerow(row)
