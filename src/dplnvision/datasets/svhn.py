from .vision import VisionDataset
import numpy as np
from PIL import Image
from pathlib import Path

from typing import (
    Any,
    Tuple,
    Optional,
    Callable
)

from utils import verify_str_arg


class SVHN(VisionDataset):
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root: Path,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.split = verify_str_arg(
            split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        self.transform = transform
        self.target_transform = target_transform

        import scipy.io as sio
        # reading mat file
        loaded_mat = sio.loadmat(root / self.filename)

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.labels[index])

        # return a PIL image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        # TODO impl
        pass
        # download_url(self.url, self.root, self.filename, self.file_md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
