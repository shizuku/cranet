import cranet
from cranet.nn import functional as F

import numpy as np


def accuracy(model, loader, loss_f=F.nll_loss):
    model.eval()
    loss = 0
    correct = 0
    with cranet.no_grad():
        for inp, lab in loader:
            out = model(inp)
            loss += loss_f(out, lab, reduction='sum').item()
            pre = out.numpy().argmax(axis=1)
            correct += (pre == lab.numpy()).sum().item()

    data_size = len(loader.dataset)
    loss /= data_size
    accu = correct / data_size
    return accu, loss


def TPR(model, loader):  # 敏感度
    TP = 0
    P = 0
    for (inp, tar) in loader:
        pre = model(inp)
        pre_am = pre.numpy().argmax(axis=1)
        TP += np.where(pre_am() == 1 and tar.numpy() == 1, 1, 0).sum()
        P += tar.numpy().sum()
    TPR = TP / P
    return TPR


def FPR(model, loader):
    FP = 0
    N = 0
    for (inp, tar) in loader:
        pre = model(inp)
        pre_am = pre.numpy().argmax(axis=1)
        FP += np.where(pre_am == 1 and tar.numpy() == 0, 1, 0).sum()
        N += tar.numpy().size() - tar.numpy().sum()
    FPR = FP / N
    return FPR


def TNR(model, loader):  # 特异度
    return 1 - FPR(model, loader)


def BER(model, loader):
    return 1 / 2 * (FPR(model, loader) + (1 - TPR(model, loader)))


def PPV(model, loader):
    TP = 0
    P = 0
    for (inp, tar) in loader:
        pre = model(inp)
        pre_am = pre.numpy().argmax(axis=1)
        P += pre_am.sum()
        TP += np.where(pre_am() == 1 and tar.numpy() == 1, 1, 0).sum()
    PPV = TP / P
    return PPV


def NPV(model, loader):
    N = 0
    TN = 0
    for (inp, tar) in loader:
        pre = model(inp)
        pre_am = pre.numpy().argmax(axis=1)
        N += pre_am.size() - pre_am.sum()
        TN += np.where(pre_am() == 0 and tar.numpy() == 0, 1, 0).sum()
    NPV = TN / N
    return NPV
