import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def softsign_update_fn(p, grad, momentum, lr, wd, beta, eps, step):
    # apply decoupled weight decay
    if wd != 0:
        p.data.mul_(1. - lr * wd)

    # accumulate momentum
    momentum.mul_(beta).add_(grad, alpha=1. - beta)
    soft_sign = momentum / torch.sqrt(torch.square(momentum) + eps ** 2)

    # update parameters using the sign of the momentum
    p.add_(soft_sign, alpha=-lr)

# class

class SoftSign(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-5,
        weight_decay: float = 0.0,
    ):
        assert lr > 0.
        assert 0. <= beta <= 1.

        defaults = dict(
            lr = lr,
            beta = beta,
            eps=eps,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = softsign_update_fn

    @torch.no_grad()
    def step(self):

        loss = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad, lr, beta, eps, state = (
                    p.grad, group['lr'], group['beta'], group['eps'], self.state[p]
                )

                # initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['step'] = 0

                momentum = state['exp_avg']
                state['step'] += 1

                # update parameters
                self.update_fn(
                    p,
                    grad,
                    momentum,
                    lr,
                    weight_decay,
                    beta,
                    eps,
                    state['step']
                )

        return loss
