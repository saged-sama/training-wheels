import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay
        )
        super().__init__(params=params, defaults=defaults)

    # @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for param in group['params']:
                if param.grad is not None:
                    if wd != 0:
                        param.grad += wd * param.data
                    param.data -= lr * param.grad

        return loss