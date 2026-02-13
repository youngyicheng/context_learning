"""CL-bench dataset loader and preprocessing."""

from typing import Iterator, List, Optional

from datasets import load_dataset


def _to_list(messages) -> list:
    """Convert dataset messages to list of dicts."""
    if hasattr(messages, "__iter__") and not isinstance(messages, (str, dict)):
        return list(messages)
    return [messages] if messages else []


def _to_dict(metadata) -> dict:
    """Convert dataset metadata to dict."""
    if isinstance(metadata, dict):
        return metadata
    return dict(metadata) if metadata else {}


class CLBenchDataLoader:
    """Load and iterate over CL-bench dataset."""

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            split: Dataset split ('train', 'validation', or 'test').
            max_samples: Maximum number of samples to load (None = all).
            subset: HuggingFace subset name (e.g., 'default').
            cache_dir: Cache directory for downloaded dataset.
        """
        self.split = split
        self.max_samples = max_samples
        self.subset = subset
        self.cache_dir = cache_dir
        self._dataset = None

    def load(self) -> "CLBenchDataLoader":
        """Load the dataset from HuggingFace."""
        kwargs = {
            "path": "tencent/CL-bench",
            "split": self.split,
            "cache_dir": self.cache_dir,
        }
        if self.subset:
            kwargs["name"] = self.subset
        self._dataset = load_dataset(**kwargs)
        return self

    @property
    def dataset(self):
        """Lazy load dataset on first access."""
        if self._dataset is None:
            self.load()
        return self._dataset

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.max_samples is not None:
            return min(n, self.max_samples)
        return n

    def __iter__(self) -> Iterator[dict]:
        """Yield samples as dicts with messages, rubrics, metadata."""
        n = len(self)
        for i in range(n):
            row = self.dataset[i]
            sample = {
                "messages": _to_list(row.get("messages", [])),
                "rubrics": _to_list(row.get("rubrics", [])),
                "metadata": _to_dict(row.get("metadata", {})),
            }
            if not sample["metadata"].get("task_id"):
                sample["metadata"]["task_id"] = str(i)
            yield sample

    def get_batch(self, batch_size: int = 1) -> Iterator[List[dict]]:
        """Yield batches of samples."""
        batch = []
        for sample in self:
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
