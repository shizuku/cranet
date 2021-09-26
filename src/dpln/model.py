from typing import List

import numpy as np

from .layer import Layer
from .loss import Loss
from .optimizer import Optimizer
from .data import Dataset


class Model:
    def __init__(self):
        pass


class Sequential(Model):
    def __init__(self, optimizer: Optimizer, loss: Loss):
        super().__init__()
        self.l: List[Layer] = []
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = []
        self.train_accu = []
        self.test_accu = []

    def __int__(self, optimizer: Optimizer, loss: Loss, l: List[Layer]):
        super().__init__()
        self.l: List[Layer] = l
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = []
        self.train_accu = []
        self.test_accu = []

    def add(self, layer: Layer):
        self.l.append(layer)

    def forward(self, x):
        z = x
        for i in self.l:
            z = i.forward(z)
        return z

    def backward(self, delta):
        dx = delta
        for i in self.l[::-1]:
            dx = i.backward(dx)
        return dx

    def update(self):
        for i in self.l:
            i.update(self.optimizer)

    def train_step(self, dataset: Dataset, epoch: int, verbose):
        for (inp, lab), i in dataset.enumerate():
            pre = self.forward(inp)
            loss = self.loss.forward(pre, lab)
            delta = self.loss.backward(pre, lab)
            self.backward(delta)
            self.update()
            self.train_loss.append(loss)
            if verbose:
                print("Epoch: {}\tStep: {}\tLoss: {}".format(
                    epoch + 1, i + 1, loss))

    def train(self, train_dataset: Dataset, test_dataset: Dataset, epochs: int, verbose=True):
        for epoch in range(epochs):
            self.train_step(train_dataset, epoch, verbose)
            train_accu = self.accuracy(train_dataset)
            test_accu = self.accuracy(test_dataset)
            self.train_accu.append(train_accu)
            self.test_accu.append(test_accu)
            print("Epoch: {}\tTrain accuracy: {:.2f}\tTest accuracy: {:.2f}".format(
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
