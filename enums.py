from enum import auto, StrEnum

class TaskType(StrEnum):
    DATASET = auto()
    BASELINE = auto()


class SearchEngine(StrEnum):
    KAGGLE = auto()
    HF = auto()
    PwC = auto()