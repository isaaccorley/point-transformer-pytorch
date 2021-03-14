import os
import glob
import shutil
from typing import Tuple, List

import torch
import torchvision.transforms as T
from torchvision.datasets.utils import download_and_extract_archive


URLS = {
    "ModelNet10": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    "ModelNet40": "http://modelnet.cs.princeton.edu/ModelNet40.zip"
}


class ModelNetDataset(torch.utils.data.Dataset):

    path: str = ""
    url: str = ""
    ext = ".off"

    def __init__(
        self,
        root: str,
        split: str,
        transforms: T.Compose
    ) -> None:

        self.root = root
        self.split = split
        self.transforms = transforms
        self.path = os.path.join(self.root, self.path)

        os.makedirs(self.root, exist_ok=True)

        # Download dataset if necessary
        if not os.path.exists(self.path):
            self.download(root=self.root, url=self.url)

        # Get class labels and mappings
        self.classes = next(os.walk(self.path))[1]
        self.idx2class = dict(enumerate(self.classes))
        self.class2idx = {c: i for i, c in self.idx2class.items()}

        # List files and their labels
        self.files, self.labels = self.list_files(root=self.path)

    def download(self, root: str, url: str) -> None:
        download_and_extract_archive(
            url=url,
            download_root=root,
            remove_finished=True
        )
        if os.path.exists(os.path.join(self.root, "__MACOSX")):
            shutil.rmtree(os.path.join(self.root, "__MACOSX"))

    def list_files(self, root: str) -> Tuple[List[str], List[int]]:
        files, labels = [], []

        for i, c in enumerate(self.classes):
            path = os.path.join(root, c, self.split)
            pcls = glob.glob(os.path.join(path, f"*{self.ext}"))
            for f in pcls:
                files.append(f)
                labels.append(i)

        return files, labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, y = self.files[idx], self.labels[idx]
        x = self.transforms(path)
        y = torch.tensor(y, dtype=torch.long)
        x = x.t()
        return x, y


class ModelNet10(ModelNetDataset):

    path = "ModelNet10"
    url = URLS[path]


class ModelNet40(ModelNetDataset):

    path = "ModelNet40"
    url = URLS[path]
