"""Reference datasets (boundary, validation, test points) with an on-disk cache."""

from .cache import get_dataset
from .dataset import DatasetSpec, ReferenceDataset

__all__ = ["DatasetSpec", "ReferenceDataset", "get_dataset"]
