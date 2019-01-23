import numpy as np
import fnmatch
import os
import matplotlib.pyplot as plt


def read_file(file_name):
    a = np.load(file_name)
    return a

# data = np.array([])

data_1 = read_file('img_p16_n0-05_reg_16_gk0-05_r0-1_pool.dat')
# data_2 = read_file('p16_o16_gk0-01_r0-01.dat')
# data_3 = read_file('p16_o16_gk0-1_r0-0.dat')
# data_4 = read_file('p16_n0-1_reg_16.dat')
# data_5 = read_file('p2_n0-1_reg_8.dat')
# data_6 = read_file('p2_n0-1_reg_16.dat')

# plt.plot(np.arange(0, 20000, 250), data_5[:, 0], label="2 Peaks, 8 Outputs", alpha=0.8, lw=1)
# plt.plot(np.arange(0, 20000, 250), data_6[:, 0], label="2 Peaks, 16 Outputs", alpha=0.8, lw=1)
plt.plot(np.arange(0, 60000, 2000), data_1[:, 0], label="Reg: 0.01", alpha=0.8, lw=1, color='blue')
plt.plot(np.arange(0, 60000, 2000), data_1[:, 1], label="8 Peaks, 16 Outputs", alpha=0.8, lw=1)
# plt.plot(np.arange(0, 30000, 2000), data_3[:, 0], label="Reg: 0.0", alpha=0.8, lw=1, color='red')
# plt.plot(np.arange(0, 20000, 250), data_4[:, 0], label="16 Peaks, 16 Outputs", alpha=0.8, lw=1)
plt.plot(np.arange(0, 60000, 2000), data_1[:, 2], label="Wrong Inputs", alpha=0.8, lw=1, color='blue', ls='--')
# plt.plot(np.arange(0, 30000, 2000), data_3[:, 2], label="Wrong Inputs", alpha=0.8, lw=1, color='red', ls='--')


plt.ylabel("Regret")
plt.xlabel("Time Step")
plt.ylim([0., 3.])
# plt.xlim([0,20000])
# plt.axhline(4)
plt.legend()
plt.grid()
plt.show()
