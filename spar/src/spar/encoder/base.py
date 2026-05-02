"""EncoderClient ABC: common interface for dense embedding models (encode list[str] → ndarray)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EncoderClient(ABC):
    @abstractmethod
    def encode(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
