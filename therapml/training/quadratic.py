import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import rc
rc('animation', html='jshtml')

X = torch.linspace(-2, 2, 1000)
Y = torch.pow(X, 2)

plt.plot(X, Y)
plt.show()
plt.savefig()