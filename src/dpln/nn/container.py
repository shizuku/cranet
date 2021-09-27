from typing import List

import numpy as np

from .module import Module
from .optimizer import Optimizer
from .loss import Loss

from ..data import Dataset


class Sequential(Module):
    def __init__(self, optimizer: Optimizer, loss: Loss):
        super().__init__()
        self.l: List[Module] = []
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = []
        self.train_accu = []
        self.test_accu = []

    def add(self, layer: Module):
        self.l.append(layer)

    def forward(self, x):
        z = x
        for i in self.l:
            z = i.forward(z)
        return z

    def backward(self, delta):
        for i in self.l[::-1]:
            delta = i.backward(delta)
        return delta

    def update(self,  optimizer: Optimizer):
        for i in self.l:
            i.update(optimizer)

    def train_step(self, dataset: Dataset, epoch: int, verbose):
        for (inp, lab), i in dataset.enumerate():
            pre = self.forward(inp)
            loss = self.loss.forward(pre, lab)
            delta = self.loss.backward(pre, lab)
            self.backward(delta)
            self.update(self.optimizer)
            self.train_loss.append(loss)
            if verbose:
                print("Epoch: {}\tStep: {}\tLoss: {}".format(
                    epoch + 1, i + 1, loss))

    def train(self, train_ds: Dataset, test_ds: Dataset, epochs: int, verbose=True):
        for epoch in range(epochs):
            self.train_step(train_ds, epoch, verbose)
            train_accu = self.accuracy(train_ds)
            test_accu = self.accuracy(test_ds)
            self.train_accu.append(train_accu)
            self.test_accu.append(test_accu)
            print("Epoch: {}\tTrain accuracy: {:.4f}\tTest accuracy: {:.4f}".format(
                epoch + 1, train_accu, test_accu))

    def accuracy(self, dataset: Dataset) -> float:
        size = 0
        right = 0
        for x, y in dataset:
            p = self.forward(x)
            eq = np.where(p.argmax(axis=-1) == y.argmax(axis=-1), 1, 0)
            right += eq.sum()
            size += eq.size
        return right / size
