from __future__ import annotations
from typing import Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions


def exists(val):
    return val is not None


# update functions


# def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, pow):
#     # stepweight decay

#     p.data.mul_(1.0 - lr * wd)

#     # weight update

#     update = torch.lerp(exp_avg, grad, 1 - beta1).sign_()
#     sign_agreement = (
#         update.eq(grad.sign()).count_nonzero().to(torch.float32) / update.numel()
#     )
#     scale = (sign_agreement * 2.0 - 1.0).abs() ** pow

#     p.add_(update, alpha=-lr * scale)

#     # decay the momentum running average coefficient

#     exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, pow, sign_agreement):
    # stepweight decay

    p.data.mul_(1.0 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1.0 - beta1).sign_()
    s_a = ((update * grad) > 0).mean(dtype=torch.float32) * 2.0 - 1.0
    scale = (
        sign_agreement.clone().mul_(beta1).add(s_a, alpha=1.0 - beta1).abs().pow(pow)
    )

    p.add_(update, alpha=-lr * scale)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
    sign_agreement.mul_(beta2).add_(s_a, alpha=1.0 - beta2)


# class


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
        decoupled_weight_decay: bool = False,
        pow: float = 1.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        self._init_lr = lr
        self.pow = pow
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            raise NotImplementedError(
                "Triton support is not implemented in this version."
            )
            # from lion_pytorch.triton import update_fn as triton_update_fn

            # self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state, decoupled_wd, init_lr, pow = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                    self.decoupled_wd,
                    self._init_lr,
                    self.pow,
                )

                # maybe decoupled weight decay

                if decoupled_wd:
                    wd /= init_lr

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["sign_agreement"] = torch.tensor(0.0, device=p.device)

                exp_avg = state["exp_avg"]
                sign_agreement = state["sign_agreement"]

                self.update_fn(
                    p, grad, exp_avg, lr, wd, beta1, beta2, pow, sign_agreement
                )

        return loss
