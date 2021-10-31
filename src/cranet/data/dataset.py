from cranet import Tensor

from typing import TypeVar, Generic, Tuple, Iterator

T_co = TypeVar('T_co', covariant=True)


class Dataset(Generic[T_co]):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item) -> T_co:
        raise NotImplementedError


class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        pass

    def __getitem__(self, item):
        pass
