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