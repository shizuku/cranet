from itertools import accumulate
from numpy.random import Generator, default_rng

from cranet import Tensor

from typing import (
    List,
    Tuple,
    TypeVar,
    Generic,
    Iterable,
    Optional,
    Union,
)

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Dataset(Generic[T_co]):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item) -> T_co:
        raise NotImplementedError


class IterableDataset(Generic[T_co]):
    def __iter__(self) -> Iterable[T_co]:
        raise NotImplementedError


class Subset(Dataset[T_co]):
    """
    Subset of a dataset at specified indices.
    """
    def __init__(self, dataset: Dataset[T_co], indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx) -> Union[T_co, Dataset[T_co]]:
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(dataset: Dataset[T], lengths: Iterable[int],
                 generator: Optional[Generator] = default_rng()) -> List[Subset[T]]:
    """Randomly split a dataset into non-overlapping new datasets of given lengths."""
    if sum(lengths) != len(dataset):
        raise ValueError(f"Illegal partition {lengths} for splitting the dataset")
    rng = generator
    indices = rng.permutation(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(accumulate(lengths), lengths)]
