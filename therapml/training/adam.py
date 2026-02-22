import numpy as np
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params=params, defaults=defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            B1, B2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                p.mul_(1 - lr * wd)

                grad = p.grad
                exp_avg.mul_(B1).add_(grad, alpha=1-B1)
                exp_avg_sq.mul_(B2).addcmul_(grad, grad, value=1-B2)

                bc1 = 1 - B1 ** t
                bc2 = 1 - B2 ** t

                denom = (exp_avg_sq.sqrt()/np.sqrt(bc2)).add_(eps)
                step_size = lr / bc1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    pass

