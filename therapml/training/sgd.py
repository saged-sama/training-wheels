import torch
import numpy as np

class SGD:
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is not None:
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class SGDMoment:
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = 0.9
        self.stepcount = 0

    def step(self):
        self.stepcount += 1
        for param in self.params:
            if param.grad is not None:
                if not hasattr(param, "velocity"):
                    param.velocity = torch.zeros_like(param)

                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data

                param.velocity = self.beta * param.velocity + param.grad
                param.data -= self.lr * param.velocity

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()