import numpy as np
import matplotlib.pyplot as plt

from typing import (
    Any,
    Iterable,
    Callable,
)


def create_curve(func: Callable[[Any], Any], data_range: Iterable[int], sample_num: int):
    x = np.linspace(*data_range, num=sample_num)
    return x, func(x)


def plot_curve(model, curve):
    pred_curve = np.array([model(x).numpy() for x, _ in curve]).squeeze()
    plt.scatter(curve.x, pred_curve)
    return pred_curve
