from enum import auto, StrEnum

class EntityType(StrEnum):
    DATASET = auto()
    BASELINE = auto()


class SearchEngine(StrEnum):
    KAGGLE = auto()
    HF = auto()
    PwC = auto()