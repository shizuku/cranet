from __future__ import annotations

from cranet import Tensor

from ..parameter import Parameter

from typing import (
    Iterator,
    Iterable,
    Optional,
    List,
    Dict,
    Set,
    Union,
    Tuple,
    Callable,
)
from collections import OrderedDict


class Module:
    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._non_persistent_buffers_set: Set[str] = set()
        self._modules: Dict[str, Optional[Module]] = OrderedDict()

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""
        This is typically used to register a buffer that should not to be
        considered a model parameter. 

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.
        """
        if name == "":
            raise KeyError("key can't be empty string")
        if '.' in name:
            raise KeyError("key can't contain '.'")
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute {name} already exist")

        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """add a parameter to the module"""
        if name == "":
            raise KeyError("key can't be empty string")
        if '.' in name:
            raise KeyError("key can't contain '.'")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute {name} already exist")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cranet.nn.Parameter or None required")
        else:
            self._parameters[name] = param

    def register_module(self, name: str, module: Optional[Module]) -> None:
        if name == "":
            raise KeyError("key can't be empty string")
        if '.' in name:
            raise KeyError("key can't contain '.'")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute {name} already exist")
        self._modules[name] = module

    def get_name(self) -> str:
        return self.__class__.__name__

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_non_persistent_buffers_set' not in self.__dict__:
            self._non_persistent_buffers_set = set()

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Union[Tensor, Module]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value)}' as parameter '{name}' "
                                "(torch.nn.Parameter or None expected)")
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError("cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{type(value)}' as child module '{name}' "
                                    "(torch.nn.Module or None expected)")
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError(f"cannot assign '{type(value)}' as buffer '{name}' "
                                        "(torch.Tensor or None expected)")
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _named_members(self, get_members_fn: Callable[[Module], Iterable], prefix='', recurse=True) -> Iterator:
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        memo: Set[Parameter] = set()
        modules: List[Tuple[str, Module]] = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = module._parameters.items()
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def named_modules(self, memo: Optional[Set[Module]] = None, prefix: str = '', remove_duplicate: bool = True):
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def children(self) -> Iterator['Module']:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def __repr__(self) -> str:
        return self.__class__.__name__
