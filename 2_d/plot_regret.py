import numpy as np
import fnmatch
import os
import matplotlib.pyplot as plt


def read_file(file_name):
    a = np.load(file_name)
    return a

data = np.array([])

data = read_file('p8_n0-1_reg_16.dat')

plt.plot(np.arange(0, 20000, 250), data[:, 0], label="Learned Actions", alpha=0.8, lw=1)
plt.plot(np.arange(0, 20000, 250), data[:, 1], label="Set Actions", alpha=0.8, lw=1)


plt.ylabel("Regret")
plt.xlabel("Time Step")
# plt.ylim([-0.5, 0.5])
# plt.xlim([0,20000])
# plt.axhline(4)
plt.legend()
plt.grid()
plt.show()
