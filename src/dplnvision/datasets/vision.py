from dpln.data import Dataset

from pathlib import Path
from .utils import string_classes
from typing import (
    Any,
    List,
    Optional,
    Callable,
)


class VisionDataset(Dataset):
    _repr_indent: int = 4

    def __init__(
            self,
            root: Path,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        if isinstance(root, string_classes):
            root = Path(root)
        self.root = root

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform"):
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return ([f"{head}{lines[0]}"] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""
