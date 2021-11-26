import cranet

from ..nn.parameter import Parameter

from typing import Iterable
from collections import defaultdict


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Optimizer:
    def __init__(self, parameters: Iterable[Parameter], defaults) -> None:
        r"""Base class for all optimizers.

        .. warning::
            Parameters need to be specified as collections that have a deterministic
            ordering that is consistent between runs. Examples of objects that don't
            satisfy those properties are sets and iterators over values of dictionaries.

        Args:
            parameters (iterable): an iterable of :class:`cranet.Tensor` s or
                :class:`dict` s. Specifies what Tensors should be optimized.
            defaults: (dict): a dict containing default values of optimization
                options (used when a parameter group doesn't specify them).
        """
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(parameters)
        if len(param_groups) == 0:
            raise ValueError('optimizer got empty parameters')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.zero_grad()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    def add_param_group(self, param_group: dict):
        assert isinstance(param_group, dict), 'param_group must be dict'

        params = param_group['params']
        if isinstance(params, cranet.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, cranet.Tensor):
                raise ValueError('optimizer must optimize Tensor')

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        self.param_groups.append(param_group)
