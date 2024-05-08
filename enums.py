from strenum import StrEnum
from enum import auto

class EntityType(StrEnum):
    DATASET = auto()
    BASELINE = auto()


class SearchEngine(StrEnum):
    KAGGLE = auto()
    HF = auto()
    PwC = auto()