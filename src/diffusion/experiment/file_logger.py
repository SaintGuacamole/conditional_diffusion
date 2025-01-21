import csv
from pathlib import Path
from typing import Union, List, Dict, Any, Optional


class FileLogger:
    def __init__(
        self,
        filename: Union[str, Path],
        fieldnames: List[str],
        buffer_size: Optional[int] = None,
    ):
        self.filename = Path(filename).with_suffix('.csv')
        self.fieldnames = fieldnames
        self.buffer_size = buffer_size or 0
        self.buffer: List[Dict[str, str]] = []

        with open(filename, 'w+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()

    def log(
        self,
        metrics: Dict[str, Any],
    ):
        self.buffer.append(metrics)

        if len(self.buffer) > self.buffer_size:
            self.flush()

    def log_batch(self, metrics_list: List[Dict[str, Any]]):
        self.buffer.extend(metrics_list)

        if len(self.buffer) > self.buffer_size:
            self.flush()

    def flush(self):
        with open(self.filename, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)
        self.buffer.clear()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()