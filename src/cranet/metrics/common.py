import cranet

import numpy as np


@cranet.no_grad()
def mean_absolute_error(model, loader):
    r = 0
    for (inp, tar) in loader:
        pre = model(inp).numpy()
        r += np.abs(pre - tar.numpy()).mean()
    return r / len(loader.dataset)


@cranet.no_grad()
def root_mean_squared_error(model, loader):
    r = 0
    for (inp, tar) in loader:
        pre = model(inp).numpy()
        r += np.sqrt(((pre - tar.numpy()) ** 2).mean())
    return r / len(loader.dataset)
