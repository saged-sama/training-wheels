import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nn_blocks import ReLU, BiasLinear, SoftMax, GELU, Sigmoid
from adam import AdamW
from sgd import SGD

from matplotlib import rc
rc('animation', html='jshtml')

X = torch.linspace(-2, 2, 1000)
Y = torch.pow(X, 2)

model = nn.Sequential(
    BiasLinear(1, 10),
    GELU(),
    BiasLinear(10, 1)
)

adamW = AdamW(model.parameters(), lr=0.1)
# sgd = SGD(model.parameters(), lr=0.1)

snapshots_every_20_epoch = []
for epoch in range(5000):
    X_input = X.unsqueeze(1)

    Y_hat = model(X_input)
    
    loss = torch.mean((Y_hat.squeeze() - Y) ** 2)
    
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    loss.backward()
    
    adamW.step()
    # sgd.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            y_pred = model(X_input)
            snapshots_every_20_epoch.append(y_pred.squeeze().detach().clone())
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'b-', label='Prediction', linewidth=2)
ax.plot(X, Y, 'r--', label='Target x²', alpha=0.5, linewidth=2)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    Y_predicted = snapshots_every_20_epoch[i]
    line.set_data(X, Y_predicted)
    ax.set_title(f'Epoch {i*20} - Approximating x²', fontsize=14)
    return line,

anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(snapshots_every_20_epoch),
    interval=100,
    blit=True,
    repeat=True
)

plt.tight_layout()
plt.show()