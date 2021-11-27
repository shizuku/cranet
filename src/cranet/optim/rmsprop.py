import cranet

from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, parameters, lr=1e-2, alpha=0.99, eps=1e-8,
                 weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(parameters, defaults)

    @cranet.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with cranet.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_svg'] = cranet.zeros_like(p)
                    if momentum > 0:
                        state['momentum_buffer'] = cranet.zeros_like(p)
                    if centered:
                        state['grad_avg'] = cranet.zeros_like(p)

                state['step'] += 1

                grad = p.grad
                square_avg = state['square_svg']

                if weight_decay != 0:
                    grad = grad + weight_decay * p

                square_avg *= alpha
                square_avg += (1 - alpha) * grad * grad

                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg *= alpha
                    grad_avg += (1 - alpha) * grad
                    square_avg += -1 * grad_avg * grad_avg
                    square_avg = cranet.sqrt(square_avg)
                    square_avg = square_avg + eps
                    avg = square_avg
                else:
                    avg = cranet.sqrt(square_avg)
                    avg = avg + eps

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf *= momentum
                    buf += (grad / avg)
                    p += -lr * buf
                else:
                    p += -lr * (grad / avg)

        return loss
