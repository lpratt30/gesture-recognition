"""PyTorch dataset and dataloader helpers for saved event-window tensors."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from project_paths import EVENT_WINDOW_TENSORS_DIR


LABEL_TO_INDEX = {
    "negative": 0,
    "positive": 1,
}


class EventWindowTensorDataset(Dataset):
    """Read exported event-window `.pt` payloads from disk."""

    def __init__(self, root=EVENT_WINDOW_TENSORS_DIR, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted(path for path in self.root.iterdir() if path.is_file() and path.suffix == ".pt")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        payload = torch.load(path, map_location="cpu")

        tensor = payload["tensor"]
        metadata = payload["metadata"]
        label = payload.get("label", metadata.get("label", "unknown"))
        target = LABEL_TO_INDEX.get(label, -1)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return {
            "tensor": tensor,
            "target": target,
            "label": label,
            "metadata": metadata,
            "path": str(path),
        }


def collate_event_window_batches(batch):
    """Stack tensors and keep per-sample metadata alongside the batch."""

    tensors = torch.stack([item["tensor"] for item in batch], dim=0)
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.long)
    labels = [item["label"] for item in batch]
    metadata = [item["metadata"] for item in batch]
    paths = [item["path"] for item in batch]
    return {
        "tensors": tensors,
        "targets": targets,
        "labels": labels,
        "metadata": metadata,
        "paths": paths,
    }


def make_event_window_dataloader(root=EVENT_WINDOW_TENSORS_DIR, batch_size=8, shuffle=True, num_workers=0, transform=None):
    """Create a `torch.utils.data.DataLoader` for saved event windows."""

    dataset = EventWindowTensorDataset(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_event_window_batches,
    )


EventFrameTensorDataset = EventWindowTensorDataset
collate_event_frame_batches = collate_event_window_batches
make_event_frame_dataloader = make_event_window_dataloader
