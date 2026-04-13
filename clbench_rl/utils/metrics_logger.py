"""Step-level metrics logger that writes JSONL for later visualisation."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsLogger:
    """Append one JSON object per training step to a ``.jsonl`` file.

    Usage::

        ml = MetricsLogger("outputs/metrics.jsonl")
        for step in range(N):
            ml.log(step=step, epoch=0, j_score=0.42, loss=1.23)
        ml.close()

    The resulting file can be read by ``scripts/plot_metrics.py``.
    """

    def __init__(self, path: str | Path, flush_every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._flush_every = flush_every
        self._count = 0
        self._start = time.time()

    def log(self, *, step: int, epoch: Optional[int] = None, **metrics: Any) -> None:
        record: Dict[str, Any] = {"step": step, "wall_time": time.time() - self._start}
        if epoch is not None:
            record["epoch"] = epoch
        record.update(metrics)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
